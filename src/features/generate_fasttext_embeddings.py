import fasttext.util

fasttext.util.download_model('pt', if_exists='ignore')
 
ft = fasttext.load_model('cc.en.300.bin')