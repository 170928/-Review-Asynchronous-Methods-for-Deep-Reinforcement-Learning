# -Review-Asynchronous-Methods-for-Deep-Reinforcement-Learning
>DeepMind, MILA
>논문 리뷰 및 구현 코드 와 코드별 해석 정리

## [Abstract]
개념적으로 단순하고 가벼운 프레임 워크를 제안하는 논문입니다.  
deep neural network controller의 optimization을 위한 asynchronous gradient descent 를 사용합니다.  
이 asynchronous gradient descent는 4개의 standatrd reinforcement learning algorithm의 변화된 형태입니다.  
parallel actor-learners는 training process를 안정화 시킵니다.  
그리고, 4가지의 standard algorithm들이 neural network controller를 성공적으로 학습되도록 만들어 줍니다.  
actor-critic의 변화된 asychronous variant는 최고의 성능을 보이며, 현재 Atari domain에서 뛰어난 성능 향상을 보여줍니다.  

## [Introduction]
