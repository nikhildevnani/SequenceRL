from sequence_game.sequence_evironment import SequenceEnvironment

sequence_env = SequenceEnvironment(2, -1000, 5, 1000)

initial_state = sequence_env.reset()


obv, reward, end, info = sequence_env.step((2,4,5))

print(obv, reward, end, info)
