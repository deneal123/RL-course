def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """
    
    q_value = 0.0
    next_states = mdp.get_next_states(state, action)
    for next_state, prob in next_states.items():
        reward = mdp.get_reward(state, action, next_state)
        next_state_value = state_values[next_state]
        q_value += prob * (reward + gamma * next_state_value)
    
    return q_value
