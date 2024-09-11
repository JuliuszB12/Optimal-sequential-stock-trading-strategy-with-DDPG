## Abstract
The aim of this work is to implement the deep deterministic policy gradient DDPG algorithm, which is a deep reinforcement learning continuous control algorithm DRL-CC in order to determine the optimal sequential stock trading strategy based on all companies included in the S&P500 index using historical share prices and technical analysis indicators. The scope of the work includes a comprehensive presentation of the component algorithms of the DDPG-DRL method in the form of a deep neural network DNN, reinforcement learning RN methods and deep reinforcement learning methods DRL, characteristics of selected technical indicators and the modeling methodology used along with a description of specific model configurations and a summary of the results for the applied implementation, data and training and testing periods.  
  
Deep learning algorithms, in response to the provided information, develop rules that bind them together as a result of the algorithm's operation, and not as a result of external interference. We distinguish supervised learning such as deep neural networks, unsupervised learning and reinforcement learning. The RL reinforcement learning algorithm introduces abstract concepts such as state, action, and reward. The value of an action in the RL algorithm is understood as the expected cumulative reward resulting from the selection of a given action provided that the optimal action selection policy is continued. A deep reinforcement learning algorithm is a combination of deep neural networks and reinforcement learning to approximate the values of actions that an agent must take based on a continuous set of state values of the RL algorithm. DDPG is a variant of deep reinforcement learning algorithm that deterministically selects actions from a continuous set of values based on the given state.
  
The use of the DDPG algorithm allows to create a model estimating the structure of the optimal portfolio of stock shares on a given day. The model agent has the ability to freely buy, sell and hold a certain number of shares of companies included in the S&P500 index, and the goal is to maximize the rate of return on the initial balance of held cash. The choice of the DDPG algorithm is justified by the characteristics of the input and output data and the complexity of the entire problem. The obtained model, in the case of a risky strategy allowing the agent to invest a significant amount of funds in shares of individual companies, provides a noticeably higher rate of return than the price index of all companies considered. In the case of a strategy forcing the agent to diversify resources, the obtained results are inconclusive. The model has no problem reproducing the price index results, but above-average profit is not guaranteed.

## Environment configuration

For Windows:  
python -m venv venv  
venv/Scripts/activate.bat //In CMD  
venv/Scripts/Activate.ps1 //In Powershell  
pip install -r requirements.txt  
  
For Mac/Linux:  
python -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt  


## Files description

**thesis_PL.pdf** represents an official diploma thesis document containing a detailed description of the algorithms used and the entire project in Polish
**data_preprocess.py** downloads data from Yahoo Finance and calculates technical analysis indicators  
**actor_critic_networks.py** are the Actor and Critic DNN neural network frameworks  
**ddpg_agent.py** is the DDPG-DRL algorithm agent  
**train.py** is the procedure for training a model  
**test.py** is the procedure of a single episode of a trained model on test data  
**environment_train.py & environment_test.py** are training and testing environments for the agent  
**capm.py** is a beta capm calculation for test data
