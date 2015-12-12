from mdpv2 import DrivingMDP


def generate_time(time):
  time = time.split(':', 1)
  correct_time = int(time[0]) * 2
  time[1] = int(time[1])
  if time[1] >= 30:
    correct_time += 1
  return correct_time


print ("Creating MDP")
newMDP = DrivingMDP()
print ("Running Value iteration")
ExpectedValues = newMDP.value_iteration()
while True:
  neighborhood = str(raw_input('What neighborhood are you currently in: '))
  time = str(raw_input('Current time (HH:MM):   '))
  try:
    correct_time = generate_time(time)
  except Exception as e:
    print "ERROR: problem converting time"
    print e
    continue

  state = (neighborhood, correct_time)
  if not newMDP.validateState(state):
    print "%s is not a valid neighborhood " % (neighborhood)
    continue
  else:
    max_action_value = float("-inf")
    max_action = None
    for action in newMDP.getActions(state):
      successors = newMDP.T(state, action)
      action_value = 0
      for successor_state, prob in successors:
        if successor_state:
          action_value += ExpectedValues[successor_state] * prob
        else:
          action_value += ExpectedValues[state] * prob

      if action_value > max_action_value:
        max_action_value  =  action_value
        max_action = action
    print max_action
