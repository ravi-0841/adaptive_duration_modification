import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # train
    ## files
    parser.add_argument('--train1', default='/home/ravi/Downloads/Emo-Conv/neutral-sad/train/neutral/',
                            help="Neutral training data")
    parser.add_argument('--train2', default='/home/ravi/Downloads/Emo-Conv/neutral-sad/train/sad/',
                            help="Emotional training data")
    parser.add_argument('--train1_2', default='/home/ravi/Downloads/Emo-Conv/neutral-sad/all_below_0.5/neutral/',
                              help="Neutral training data")
    parser.add_argument('--train2_2', default='/home/ravi/Downloads/Emo-Conv/neutral-sad/all_below_0.5/sad/',
                              help="Emotional training data")
    parser.add_argument('--train1_cmu', default='/home/ravi/Desktop/pytorch-speech-transformer/data/CMU-ARCTIC/train/source/',
                            help="CMU m1-f1 training data")
    parser.add_argument('--train2_cmu', default='/home/ravi/Desktop/pytorch-speech-transformer/data/CMU-ARCTIC/train/target/',
                            help="CMU m2-f2 training data")
    parser.add_argument('--valid1', default='/home/ravi/Downloads/Emo-Conv/neutral-sad/valid/neutral/',
                             help="Neutral validation data")
    parser.add_argument('--valid2', default='/home/ravi/Downloads/Emo-Conv/neutral-sad/valid/sad/',
                             help="Emotional validation data")
    parser.add_argument('--valid1_cmu', default='/home/ravi/Desktop/pytorch-speech-transformer/data/CMU-ARCTIC/valid/source/',
                            help="CMU m1-f1 validation data")
    parser.add_argument('--valid2_cmu', default='/home/ravi/Desktop/pytorch-speech-transformer/data/CMU-ARCTIC/valid/target/',
                            help="CMU m2-f2 validation data")
    parser.add_argument('--test1_cmu', default='/home/ravi/Desktop/pytorch-speech-transformer/data/CMU-ARCTIC/test/source/',
                            help="CMU m1-f1 testing data")
    parser.add_argument('--test2_cmu', default='/home/ravi/Desktop/pytorch-speech-transformer/data/CMU-ARCTIC/test/target/',
                            help="CMU m2-f2 testing data")

    # training scheme
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--eval_batch_size', default=1, type=int)
    parser.add_argument('--num_epochs', default=2500, type=int)
    parser.add_argument('--pad_signature', default=10, type=int)
    parser.add_argument('--parallel_steps', default=1, type=int) #1
    

    parser.add_argument('--lr', default=0.001, type=float, help="learning rate") #0.0003
    parser.add_argument('--tradeoff', default=0.1, type=float, 
                        help="hyperparameter for cross-entropy loss") #1
    parser.add_argument('--warmup_steps', default=30000, type=int) #40000
    parser.add_argument('--logdir', default="log/1", help="log directory")
    parser.add_argument('--evaldir', default="eval/1", help="evaluation dir")

    # model
    parser.add_argument('--hop_size', default=0.005, type=float,
                        help="Stride size of feature window (in sec)") #0.01
    parser.add_argument('--win_size', default=0.005, type=float,
                        help="Window size of feature window (in sec)") #0.01
    parser.add_argument('--d_in', default=80, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_model', default=64, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=256, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_encoder_blocks', default=3, type=int,
                        help="number of encoder blocks") #6
    parser.add_argument('--num_decoder_blocks', default=2, type=int,
                        help="number of decoder blocks") #2
    parser.add_argument('--num_heads', default=1, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=350, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=350, type=int,
                        help="maximum length of a target sequence") # make sure it is a multiple of parallel_steps
    parser.add_argument('--dropout_rate', default=0.0, type=float) #0.2
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # test
    parser.add_argument('--test1', default='/home/ravi/Downloads/Emo-Conv/neutral-sad/test/neutral/',
                        help="Neutral test data")
    parser.add_argument('--test2', default='/home/ravi/Downloads/Emo-Conv/neutral-sad/test/sad/',
                        help="Emotional test data")
    parser.add_argument('--ckpt', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=4, type=int)
    parser.add_argument('--testdir', default="test/1", help="test result dir")
