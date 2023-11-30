import gymnasium as gym
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque
from copy import deepcopy

class PrioritizedReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(size=buffer_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs].to(device()),
            self.action[sample_idxs].to(device()),
            self.reward[sample_idxs].to(device()),
            self.next_state[sample_idxs].to(device()),
            self.done[sample_idxs].to(device())
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


class Experience_replay_buffer:
    def __init__(self,memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.buffer = namedtuple('buffer',
                                 field_names = ['state','action','reward','next_state','done'])
        self.memory = deque(maxlen=self.memory_size)
        
    def append(self,s_0,a,r,d,s_1):
        self.memory.append(self.buffer(s_0,a,r,d,s_1))
        
    def sample(self,batch_size):
        samples = np.random.choice(len(self.memory),batch_size,replace=False)
        
        batch = zip(*[self.memory[i] for i in samples])
        return batch
    def burn_in_capacity(self):
        return len(self.memory) / self.burn_in
    def capacity(self):
        return len(self.memory) / self.memory_size
    
class DQN(nn.Module):
    def __init__(self,env,activation = F.relu(), device=torch.device('cpu')):
        super(DQN, self).__init__()
        self.env = env
        self.device = device
        self.conv1 = nn.Conv2d(env.observation_space.shape[0], 16, kernel_size=8, stride=4)  # [N, 4, 84, 84] -> [N, 16, 20, 20]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 20, 20] -> [N, 32, 9, 9]
        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, env.action_space.n)
        self.activation = activation
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = x.view(-1, self.in_features)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
    
class Policy(nn.Module):
    continuous = False # you can change this
    
    def __init__(self, env, device =torch.device('cpu')):
        #itilize netwroks, memory, optimizer, loss criterion and parameters here
        super(Policy, self).__init__() # initialize super class
        self.device = device # device to run the code on
        self.env = env 
        self.current_net = DQN(env,device=device) # initialize current network
        self.target_net = DQN(env,device=device) # initialize target network
        self.memory = PrioritizedReplayBuffer(env.observation_space.shape[0], env.action_space.n, 10000)
        self.learning_rate = 0.0001
        self.optimizer = torch.optim.Adam(self.current_net.parameters(), lr=self.learning_rate)
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 50000
        self.batch_size = 32
        self.gamma = 0.99
        
    def forward(self, x):
        x = self.current_net(x)
        return x
    
    def act(self, state, training = True):
        #TODO: implement epsilon greedy policy here (random action or argmax of Q(s,a))
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        if training and np.random.rand() < epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            return self.current_net(state).max(1)[1].view(1, 1)

    def train(self):
        # TODO
        # 1. sample batch from memory
        if len(self.memory) < self.batch_size:
            return
        state , action , reward , next_state , done , indices , weights = self.memory.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        # 2. get the q values for current state and next state from current network and target network respectively
        q_values = self.current_net(state) # compute q values for current state
        next_q_values = self.current_net(next_state) # compute q values for next state
        next_q_state_values = self.target_net(next_state) # compute q values for next state from target network
        # 3. compute the expected q values
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) # gather the q values of the actions taken
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1) # gather the q values of the actions taken
        expected_q_value = reward + self.gamma * next_q_value * (1 - done) # compute the expected q value
        
        loss = (q_value - expected_q_value.detach()).pow(2) * weights # huber loss
        prios = loss + 1e-5 # small epsilon to avoid zero priority
        loss = loss.mean() # mean loss over batch
        
        self.optimizer.zero_grad() # zero gradients from previous step
        loss.backward() # compute gradients
        self.memory.update_priorities(indices, prios.data.cpu().numpy()) # update priorities
        self.optimizer.step() # apply gradients
        if self.steps_done % 1000 == 0:
            self.update_target() # update target network every 1000 steps
        
        return
    
    def update_target(self):
        self.target_net.load_state_dict(self.current_net.state_dict())

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt'), map_location=self.device)

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
