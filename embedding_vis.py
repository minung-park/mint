import os
import tensorflow as tf
import pickle
from tensorflow.contrib.tensorboard.plugins import projector


def make_metafile(vocab_file, metadata_path):
    vocabs = pickle.load(open(vocab_file,'rb'))
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        for row in vocabs:
            metadata_file.write('%s\n' % (row))
            

def draw(LOG_DIR, file_list):
    #LOG_DIR = os.getcwd()+ '/logs'
    #file_list = ['1_wv','2_wv','3_wv','9_wv']
    embeds_var = []
    tf_config = tf.ConfigProto(device_count={'GPU':0})
    config = projector.ProjectorConfig()
    
    
    with tf.Session(config=tf_config) as sess:
        for file_name in file_list:
            embed_file = './word-vectors/'+file_name+'_embedding.p'
            embed = pickle.load(open(embed_file,'rb'))
            embed_tensor = tf.Variable(embed, name=file_name+'_embeds')
            embeds_var.append(embed_tensor)
        
        
            #saver = tf.train.Saver([embeds])
            sess.run(embed_tensor.initializer)
            embedding = config.embeddings.add()
            embedding.tensor_name = embed_tensor.name
            embedding.metadata_path = os.getcwd()+'/word-vectors/'+file_name+'_metadata.tsv'
            
            projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
            
            #saver = tf.train.Saver([embed_tensor])
            #saver.save(sess, LOG_DIR+'/'+file_name+'_embeds.ckpt')
        
        result = sess.run(embeds_var)
        saver = tf.train.Saver(embeds_var)
        saver.save(sess, LOG_DIR+'/embeds.ckpt')

        
if __name__ == "__main__":
    
    LOG_DIR = os.getcwd()+ '/logs'
    file_list = ['nbt_wiki_mecab']
    #for file_name in file_list:
    #    metadata_path = os.getcwd()+'/data/'+file_name+'_metadata.tsv'
    #    vocab_file = './data/'+file_name+'_vocab.p'
    #    make_metafile(vocab_file, metadata_path)

    draw(LOG_DIR, file_list)
