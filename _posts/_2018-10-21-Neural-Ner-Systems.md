Sequence Labelling
==================

"Bidirectional LSTM-CRF Models for Sequence Tagging" (2015)
paper: https://arxiv.org/pdf/1508.01991v1.pdf

- no char-level Embeddings
- combined neural networks with hand-crafted features

- - - - - - - - - - - - - - - - - - - - - - - - - - - -

"Named Entity Recognition with Bidirectional LSTM-CNNs" (2016)
paper: https://www.aclweb.org/anthology/Q16-1026

- do not uses CRF on top
- external knowledge: char-type, capitalization and lexical features, NER
  specific processing, replacing all sequence of digits 0-9 by 0

- - - - - - - - - - - - - - - - - - - - - - - - - - - -


"Neural Architectures for Named Entity Recognition" (2016)
paper: http://www.aclweb.org/anthology/N16-1030
code:  https://github.com/Hironsan/anago
       https://github.com/achernodub/bilstm-cnn-crf-tagger
       https://github.com/glample/tagger

- use LSTM for char embeddings instead of CNN

- - - - - - - - - - - - - - - - - - - - - - - - - - - -


"End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF" (2016)
paper: http://www.aclweb.org/anthology/P16-1101
code:  https://github.com/achernodub/bilstm-cnn-crf-tagger

1) For each word, char-level representation is computed by the CNN with char embeddings as input
2) Concatenate char-level representation of word with the word embedding (pre-trained)
3) Fed into a bi-LSTM
4) Fed output vectors of bi-LSTM to CRF layer


Embeddings
==========

"Deep contextualized word representations" (2018)
paper: http://aclweb.org/anthology/N18-1202
code:  https://github.com/Hironsan/anago


"Enriching Word Vectors with Subword Information"
paper: http://aclweb.org/anthology/Q17-1010
code:  https://github.com/facebookresearch/fastText



https://createmomo.github.io/2017/09/12/CRF_Layer_on_the_Top_of_BiLSTM_1/