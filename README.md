# MINT
기존 nbt 에서 임베딩을 BERT 추가하여 실험함  
  
  
## NBT
An implementation of the Fully Data-Driven version of the Neural Belief Tracking (NBT) model (ACL 2018, [Fully Statistical Neural Belief Tracking](https://arxiv.org/abs/1805.11350)).  
This version of the model uses a learned belief state update in place of the rule-based mechanism used in the original paper. Requests are not a focus of this paper and should be ignored in the output.  

### Configuring the Tool

The config file in the config directory specifies the model hyperparameters, training details, dataset, ontologies, etc. 

### Running Experiments

train.sh and test.sh can be used to train and test the model (using the default config file). 
track.sh uses the trained models to 'simulate' a conversation where the developer can enter sequential user turns and observe the change in belief state.  
  
  
## BERT
BERT MODEL: BERT-Base, Uncased  
[Bidirectional Encoder Representations from Transformers](https://github.com/google-research/bert)

## Experiment
단어/문장 두 레벨로 나누어 테스트  
문장 단위 임베딩은 [bert-as-service](https://github.com/hanxiao/bert-as-service) 를 사용하여 진행함 
  
코드에 768 붙어있는 것들이 bert version 코드  
현재는 문장단위로 실험했던 코드들이 남아있음  
  
jupyter notebook 으로 embedding visualization 확인 가능  
