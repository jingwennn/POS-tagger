from nltk.corpus import brown
import numpy as np
import pandas as pd

# extract unique words and unique tags from training corpus
def split_training_corpus(corpus):
    words=[]
    tags=[]
    for i in range(len(corpus)):
        words.append(corpus[i][0])
        tags.append(corpus[i][1])
    words=np.unique(words)
    tags=np.unique(tags) 
    return list(words), list(tags)

# generate initial_states, transition_matrix, emission_matrix
# with training corpus
def generate_matrix(corpus, words, tags, smooth_parameter):
    # n: the number of unique words
    # m: the number of tags
    n=len(words)
    m=len(tags)
    
    # generate initial states
    initial_states=smooth_parameter*np.ones(m)
    
    # use the first words in each sentence
    initial_states[tags.index(corpus[0][1])]=initial_states[tags.index(corpus[0][1])]+1

    for i in range(1,len(corpus)):
        if corpus[i-1][0]=='.' and corpus[i][0].isalpha():
            initial_states[tags.index(corpus[i][1])]=initial_states[tags.index(corpus[i][1])]+1
                
    initial_states=initial_states/np.sum(initial_states)  
    
    # generate the transition matrix
    transition_matrix=smooth_parameter*np.ones((m,m))
    for i in range(1,len(corpus)):
        transition_matrix[tags.index(corpus[i-1][1])][tags.index(corpus[i][1])]=transition_matrix[tags.index(corpus[i-1][1])][tags.index(corpus[i][1])]+1
    
    transition_matrix=(transition_matrix.T/np.sum(transition_matrix,axis=1)).T
    
    # generate emission matrix
    emission_matrix=smooth_parameter*np.ones((m,n+1))
    for i in range(len(corpus)):
        emission_matrix[tags.index(corpus[i][1])][words.index(corpus[i][0])]=emission_matrix[tags.index(corpus[i][1])][words.index(corpus[i][0])]+1
    
    emission_matrix=(emission_matrix.T/np.sum(emission_matrix,axis=1)).T
    
    return initial_states, transition_matrix, emission_matrix
    
    
# Viterbi Algorithm from hw08
def infer_states(obs, pi, A, B):    
    state_obs_lik = np.array([B[:,z] for z in obs]).T
    
    N = state_obs_lik.shape[0] # len of state graph
    T = state_obs_lik.shape[1] # len of observations

    viterbi = np.zeros((N, T))
    back_pointer = np.zeros((N, T), dtype=int) 
    best_path = np.zeros(T, dtype=int) 
    
    # initialization step
    viterbi[:,0] = pi + state_obs_lik[:,0] 
    back_pointer[:,0] = 0 
    
    # recursion step
    for t in range(1, T):
        for s in range(N):
            viterbi[s, t] = np.amax(viterbi[:,t-1] + A[:,s] + state_obs_lik[s,t])
            back_pointer[s, t] = np.argmax(viterbi[:,t-1] + A[:,s] + state_obs_lik[s,t])

    # termination step        
    best_path_prob = np.amax(viterbi[:,-1]) 
    best_path_pointer =  np.argmax(viterbi[:,-1])
    
    best_path[-1] = best_path_pointer
    for t in range(T-2, -1, -1):
        best_path[t] = back_pointer[(best_path[t+1]), t+1]

    return list(best_path), best_path_prob   
 
# split the testing corpus  
def split_testing_corpus(corpus, train_words, train_tags):
    test_words=[]
    test_tags=[]
    for i in range(len(corpus)):
        if corpus[i][0] in train_words:
            test_words.append(train_words.index(corpus[i][0]))
        else: 
            test_words.append(len(train_words))
        test_tags.append(train_tags.index(corpus[i][1]))
        
    return test_words, test_tags

# compute the confusion matrix
def compute_confusion_matrix(true_tags, tags, m):
    confusion_matrix=np.zeros((m,m))
    for i in range(len(true_tags)):
        confusion_matrix[true_tags[i]][tags[i]]=confusion_matrix[true_tags[i]][tags[i]]+1

    return confusion_matrix

# training corpus and testing corpus
corpus_fiction = brown.tagged_words(categories='fiction', tagset='universal')
corpus_science = brown.tagged_words(categories='science_fiction', tagset='universal')
corpus_reviews = brown.tagged_words(categories='reviews', tagset='universal')
corpus_hobbies = brown.tagged_words(categories='hobbies', tagset='universal')

# get words and tags from training corpus
words_train_fiction,tags_train_fiction=split_training_corpus(corpus_fiction)
# train HMM
initial_states_train, transition_matrix_train, emission_matrix_train=generate_matrix(corpus_fiction, words_train_fiction, tags_train_fiction,1)
# get words and tags from testing corpus
words_test_science, tags_test_science=split_testing_corpus(corpus_science, words_train_fiction, tags_train_fiction)
words_test_reviews, tags_test_reviews=split_testing_corpus(corpus_reviews, words_train_fiction, tags_train_fiction)
words_test_hobbies, tags_test_hobbies=split_testing_corpus(corpus_hobbies, words_train_fiction, tags_train_fiction)
# infer the tags
tags_infer_science, prob1=infer_states(words_test_science, np.log(initial_states_train), np.log(transition_matrix_train), np.log(emission_matrix_train))
tags_infer_reviews, prob2=infer_states(words_test_reviews, np.log(initial_states_train), np.log(transition_matrix_train), np.log(emission_matrix_train))
tags_infer_hobbies, prob3=infer_states(words_test_hobbies, np.log(initial_states_train), np.log(transition_matrix_train), np.log(emission_matrix_train))
# compute accuracy
print("Accuracy for science fiction:")
print(np.sum(np.array(tags_infer_science)==np.array(tags_test_science))/len(tags_test_science))

print("Accuracy for reviews:")
print(np.sum(np.array(tags_infer_reviews)==np.array(tags_test_reviews))/len(tags_test_reviews))

print("Accuracy for hobbies:")
print(np.sum(np.array(tags_infer_hobbies)==np.array(tags_test_hobbies))/len(tags_test_hobbies))

# compute confusion matrix
confusion_matrix_science=compute_confusion_matrix(tags_test_science, tags_infer_science, len(tags_train_fiction))
confusion_matrix_reviews=compute_confusion_matrix(tags_test_reviews, tags_infer_reviews, len(tags_train_fiction))
confusion_matrix_hobbies=compute_confusion_matrix(tags_test_hobbies, tags_infer_hobbies, len(tags_train_fiction))

# plot confusion matrix
df1 = pd.DataFrame(confusion_matrix_science.astype(int), tags_train_fiction)
df1.columns = tags_train_fiction
df1.name = 'Training corpus: fiction     Testing corpus: science fiction'
print("Training corpus: fiction     Testing corpus: science fiction")
print(df1)

df2 = pd.DataFrame(confusion_matrix_reviews.astype(int), tags_train_fiction)
df2.columns = tags_train_fiction
df2.name = 'Training corpus: fiction     Testing corpus: reviews'
print("Training corpus: fiction     Testing corpus: reviews")
print(df2)

df3 = pd.DataFrame(confusion_matrix_hobbies.astype(int), tags_train_fiction)
df3.columns = tags_train_fiction
df3.name = 'Training corpus: fiction     Testing corpus: hobbies'
print("Training corpus: fiction     Testing corpus: hobbies")
print(df3)

# compute error rate for each tag
error1=(np.sum(confusion_matrix_science,axis=1)-np.diag(confusion_matrix_science))/np.sum(confusion_matrix_science,axis=1)
print("error rate for Science Fiction")
print(error1)
error2=(np.sum(confusion_matrix_reviews,axis=1)-np.diag(confusion_matrix_reviews))/np.sum(confusion_matrix_reviews,axis=1)
print("error rate for Reviews")
print(error2)
error3=(np.sum(confusion_matrix_hobbies,axis=1)-np.diag(confusion_matrix_hobbies))/np.sum(confusion_matrix_hobbies,axis=1)
print("error rate for Hobbies")
print(error3)