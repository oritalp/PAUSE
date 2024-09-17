# PAUSE: Privacy-Aware User SElection for Federated Learning
The code for the paper's simulations is attached, clone the __production__ branch and run the following code snippets:

#### To run Test 1 - small scale network test on truncated MNIST (choosing 5 out of 30 users at each round), run the following command:

python main.py --data mnist --data_truncation 2000 --model mlp --num_users 30 --num_users_per_round 5 --epsilon_bar 200 --delta_f 0.003 --alpha 100 --beta 2 --gamma 5


#### To run Test 2 - small sacle network test on CIFAR-10 (choosing 5 out of 30 users at each round), run the following command:


python main.py --data cifar10 --data_truncation None --model cnn3 --num_users 30 --num_users_per_round 5 --epsilon_bar 100 --delta_f 0.012 --alpha 100 --beta 2 --gamma 5
