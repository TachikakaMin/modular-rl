from __future__ import print_function
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import xmltodict
import wrappers
import gym
from gym.envs.registration import register
from shutil import copyfile
from config import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import subprocess

def makeEnvWrapper(env_name, xml, max_episode_steps, env_file, obs_max_len=None, seed=0, vis=False):
    """return wrapped gym environment for parallel sample collection (vectorized environments)"""
    def helper():
        params = {'xml': os.path.abspath(xml)}
        if not vis:
            register(id=("%s-v0" % env_name),
                    max_episode_steps=max_episode_steps,
                    entry_point="environments.%s:ModularEnv" % env_file,
                    kwargs=params)
        e = gym.make("environments:%s-v0" % env_name)
        e.seed(seed)
        return wrappers.ModularEnvWrapper(e, obs_max_len)
    return helper


def findMaxChildren(env_names, graphs):
    """return the maximum number of children given a list of env names and their corresponding graph structures"""
    max_children = 0
    for name in env_names:
        most_frequent = max(graphs[name], key=graphs[name].count)
        max_children = max(max_children, graphs[name].count(most_frequent))
    return max_children


def registerEnvs(env_names, args, vis=False):
    """register the MuJoCo envs with Gym and return the per-limb observation size and max action value (for modular policy training)"""
    max_episode_steps = args.max_episode_steps
    custom_xml = args.custom_xml
    # get all paths to xmls (handle the case where the given path is a directory containing multiple xml files)
    paths_to_register = []
    # existing envs
    if not custom_xml:
        for name in env_names:
            paths_to_register.append(os.path.join(XML_DIR, "{}.xml".format(name)))
    # custom envs
    else:
        if os.path.isfile(custom_xml):
            paths_to_register.append(custom_xml)
        elif os.path.isdir(custom_xml):
            for name in sorted(os.listdir(custom_xml)):
                if '.xml' in name:
                    paths_to_register.append(os.path.join(custom_xml, name))

    envs_train = []
    # register each env
    for xml in paths_to_register:
        env_name = os.path.basename(xml)[:-4]
        env_file = env_name
        # create a copy of modular environment for custom xml model
        if not os.path.exists(os.path.join(ENV_DIR, '{}.py'.format(env_name))):
            # create a duplicate of gym environment file for each env (necessary for avoiding bug in gym)
            copyfile(BASE_MODULAR_ENV_PATH, '{}.py'.format(os.path.join(ENV_DIR, env_name)))
        params = {'xml': os.path.abspath(xml)}
        # register with gym
        register(id=("%s-v0" % env_name),
                 max_episode_steps=max_episode_steps,
                 entry_point="environments.%s:ModularEnv" % env_file,
                 kwargs=params)
        
        env = gym.make("environments:%s-v0" % env_name)
        env = wrappers.IdentityWrapper(env)
        # the following is the same for each env
        limb_obs_size = env.limb_obs_size
        max_action = env.max_action
        obs_max_len = max([len(args.graphs[env_name]) for env_name in env_names]) * limb_obs_size
        envs_train.append(makeEnvWrapper(env_name, xml, max_episode_steps, env_file,  obs_max_len, args.seed, vis))
    return limb_obs_size, max_action, envs_train



def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
    Args
    q: 1x4 quaternion
    Returns
    r: 1x3 exponential map
    Raises
    ValueError if the l2 norm of the quaternion is not close to 1
    """
    if (np.abs(np.linalg.norm(q)-1)>1e-3):
        raise(ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]
    r0 = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
    theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
    theta = np.mod( theta + 2*np.pi, 2*np.pi )
    if theta > np.pi:
        theta =  2 * np.pi - theta
        r0    = -r0
    r = r0 * theta
    return r

# replay buffer: expects tuples of (state, next_state, action, reward, done)
# modified from https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class ReplayBuffer(object):
    def __init__(self, max_size=1e6, slicing_size=None):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        # maintains slicing info for [obs, new_obs, action, reward, done]
        if slicing_size:
            self.slicing_size = slicing_size
        else:
            self.slicing_size = None

    def add(self, data):
        if self.slicing_size is None:
            self.slicing_size = [data[0].size, data[1].size, data[2].size, 1, 1]
        data = np.concatenate([data[0], data[1], data[2], [data[3]], [data[4]]])
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            data = self.storage[i]
            X = data[:self.slicing_size[0]]
            Y = data[self.slicing_size[0]:self.slicing_size[0] + self.slicing_size[1]]
            U = data[self.slicing_size[0] + self.slicing_size[1]:self.slicing_size[0] + self.slicing_size[1] + self.slicing_size[2]]
            R = data[self.slicing_size[0] + self.slicing_size[1] + self.slicing_size[2]:self.slicing_size[0] + self.slicing_size[1] + self.slicing_size[2] + self.slicing_size[3]]
            D = data[self.slicing_size[0] + self.slicing_size[1] + self.slicing_size[2] + self.slicing_size[3]:]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return (np.array(x), np.array(y), np.array(u),
                    np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1))

    def sample_seq_len(self, batch_size, seq_len):
        ind = np.random.randint(0, len(self.storage)-seq_len, size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            x1, y1, u1, r1, d1 = [], [], [], [], []
            for j in range(i,i+seq_len):
                data = self.storage[j]
                X = data[:self.slicing_size[0]]
                Y = data[self.slicing_size[0]:self.slicing_size[0] + self.slicing_size[1]]
                U = data[self.slicing_size[0] + self.slicing_size[1]:self.slicing_size[0] + self.slicing_size[1] + self.slicing_size[2]]
                R = data[self.slicing_size[0] + self.slicing_size[1] + self.slicing_size[2]:self.slicing_size[0] + self.slicing_size[1] + self.slicing_size[2] + self.slicing_size[3]]
                D = data[self.slicing_size[0] + self.slicing_size[1] + self.slicing_size[2] + self.slicing_size[3]:]
                x1.append(np.array(X, copy=False))
                y1.append(np.array(Y, copy=False))
                u1.append(np.array(U, copy=False))
                r1.append(np.array(R, copy=False))
                d1.append(np.array(D, copy=False))
            x.append(np.array(x1, copy=False))
            y.append(np.array(y1, copy=False))
            u.append(np.array(u1, copy=False))
            r.append(np.array(r1, copy=False))
            d.append(np.array(d1, copy=False))

        return (np.array(x), np.array(y), np.array(u),
                    np.array(r), np.array(d))


class MLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MLPBase, self).__init__()
        self.l1 = nn.Linear(num_inputs, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, num_outputs)

    def forward(self, inputs):
        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


def getGraphStructure(xml_file):
    """Traverse the given xml file as a tree by pre-order and return the graph structure as a parents list"""
    def preorder(b, parent_idx=-1):
        self_idx = len(parents)
        parents.append(parent_idx)
        if 'body' not in b:
            return
        if not isinstance(b['body'], list):
            b['body'] = [b['body']]
        for branch in b['body']:
            preorder(branch, self_idx)
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    parents = []
    try:
        root = xml['mujoco']['worldbody']['body']
        assert not isinstance(root, list), 'worldbody can only contain one body (torso) for the current implementation, but found {}'.format(root)
    except:
        raise Exception("The given xml file does not follow the standard MuJoCo format.")
    preorder(root)
    # signal message flipping for flipped walker morphologies
    if 'walker' in os.path.basename(xml_file) and 'flipped' in os.path.basename(xml_file):
        parents[0] = -2
    return parents


def getGraphJoints(xml_file):
    """Traverse the given xml file as a tree by pre-order and return all the joints defined as a list of tuples (body_name, joint_name1, ...) for each body"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""
    def preorder(b):
        if 'joint' in b:
            if isinstance(b['joint'], list) and b['@name'] != 'torso':
                raise Exception("The given xml file does not follow the standard MuJoCo format.")
            elif not isinstance(b['joint'], list):
                b['joint'] = [b['joint']]
            joints.append([b['@name']])
            for j in b['joint']:
                joints[-1].append(j['@name'])
        if 'body' not in b:
            return
        if not isinstance(b['body'], list):
            b['body'] = [b['body']]
        for branch in b['body']:
            preorder(branch)
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    try:
        root = xml['mujoco']['worldbody']['body']
    except:
        raise Exception("The given xml file does not follow the standard MuJoCo format.")
    preorder(root)
    return joints


def getMotorJoints(xml_file):
    """Traverse the given xml file as a tree by pre-order and return the joint names in the order of defined actuators"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    motors = xml['mujoco']['actuator']['motor']
    if not isinstance(motors, list):
        motors = [motors]
    for m in motors:
        joints.append(m['@joint'])
    return joints


# Print iterations progress
# from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar(iteration, total, prefix='Video Progress:', suffix='Complete', decimals=1, length=35, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def plot_video(policy, args, env_name, env):
    env = makeEnvWrapper(env_name, seed=args.seed, obs_max_len=None)()
    total_time = args.video_length * 100
    policy.change_morphology(args.graphs[env_name])

    # create unique temp frame dir
    count = 0
    frame_dir = os.path.join(VIDEO_DIR, "frames_{}_{}_{}".format(args.expID, env_name, count))
    while os.path.exists(frame_dir):
        count += 1
        frame_dir = "{}/frames_{}_{}_{}".format(VIDEO_DIR, args.expID, env_name, count)
    os.makedirs(frame_dir)
    # create video name without overwriting previously generated videos
    count = 0
    video_name = "%04d_%s_%d" % (args.expID, env_name, count)
    while os.path.exists("{}/{}.mp4".format(VIDEO_DIR, video_name)):
        count += 1
        video_name = "%04d_%s_%d" % (args.expID, env_name, count)

    # init env vars
    done = True
    print("-" * 50)
    time_step_counter = 0
    printProgressBar(0, total_time)

    while time_step_counter < total_time:
        printProgressBar(time_step_counter + 1, total_time, prefix=env_name)
        if done:
            obs = env.reset()
            done = False
            episode_reward = 0
        action = policy.select_action(np.array(obs))
        # perform action in the environment
        new_obs, reward, done, _ = env.step(action)
        episode_reward += reward
        # draw image of current frame
        image_data = env.sim.render(VIDEO_RESOLUATION[0], VIDEO_RESOLUATION[1], camera_name="track")
        img = Image.fromarray(image_data, "RGB")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('./misc/sans-serif.ttf', 24)
        draw.text((200, 10), "Instant Reward: " + str(reward), (255, 0, 0), font=font)
        draw.text((200, 35), "Episode Reward: " + str(episode_reward), (255, 0, 0), font=font)
        img.save(os.path.join(frame_dir, "frame-%.10d.png" % time_step_counter))

        obs = new_obs
        time_step_counter += 1

    # redirect output so output does not show on window
    FNULL = open(os.devnull, 'w')
    # create video
    subprocess.call(['ffmpeg', '-framerate', '50', '-y', '-i', os.path.join(frame_dir, 'frame-%010d.png'),
                        '-r', '30', '-pix_fmt', 'yuv420p', os.path.join(VIDEO_DIR, '{}.mp4'.format(video_name))],
                        stdout=FNULL, stderr=subprocess.STDOUT)
    subprocess.call(['rm', '-rf', frame_dir])
