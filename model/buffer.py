class Buffer():
    def __init__(self) -> None:
        self.states = []
        self.rewards = []
        self.terminates = []
        self.logprobs = []
        self.actions = []
    
    def clear(self):
        del self.states[:]
        del self.rewards[:]
        del self.terminates[:]
        del self.logprobs[:]
        del self.actions[:]
    
    def add(self,state,reward,terminate,logprob,action):
        self.states.append(state)
        self.rewards.append(reward)
        self.terminates.append(terminate)
        self.logprobs.append(logprob)
        self.actions.append(action)
