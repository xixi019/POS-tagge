from tagger import pre_process, model
import argparse
import yaml
import os
from nltk import tokenize
import sklearn_crfsuite
from sklearn.metrics import classification_report


def run(args):
    mode = args.mode
    text = args.text
    gold = args.gold
    with open(args.config, "r") as yamlin:
        config = yaml.safe_load(yamlin)
    if mode == "train":
        print(f"Training a model on {config['train_file']}.")
        X, Y = pre_process.load_dataset(config["train_file"])
        X, Y = pre_process.prepare_data_for_training_crf(X, Y)
        crf = model.fit_and_report_crf(X, Y,
                                       config["crossval"], config["n_folds"])
        model.save_model_crf(crf, config["model_file"])

    elif mode == "tag":
        # creat a file name for saving the result file.
        text_dir = os.path.abspath('.').replace('\\', '/')+"/"+text
        save_path = text_dir + ".tag"
        if os.path.isfile(text_dir) is True:
            print(f"Tagging file using pretrained model: {config['train_file']}.")
            with open(text_dir, 'r') as f:
                text = f.read()
                text = tokenize.sent_tokenize(text)
                crf = model.load_model_crf(config["model_file"])
                with open(save_path, "w") as tag_file:
                    for sentence in text:
                        tagged_sent = model.tag_sentence_crf(sentence, crf)
                        print(tagged_sent)
                        model.print_tagged_sent_crf(tagged_sent)
                        print()  # Add a new line when a sentence ends.
                        toFile = model.return_tagged_sent_crf(tagged_sent)
                        for line in toFile:
                            tag_file.write(line)
                            tag_file.write("\n")
                tag_file.close()
        else:
            print(f"Tagging text using pretrained model: {config['train_file']}.")
            crf = model.load_model_crf(config["model_file"])
            tagged_sent = model.tag_sentence_crf(args.text, crf)
            model.print_tagged_sent_crf(tagged_sent)

    elif mode == "eval":
        y_pred = []
        y_true = []
        # load the date and output two list 
        # which contain sents and tags.
        sents_to_tag, label_ls = pre_process.load_dataset(gold)
        for ls in label_ls:
            for label in ls:
                y_true.append(label)
        crf = model.load_model_crf(config["model_file"])
        # tag the corpus sentence by sentence.
        for sentence in sents_to_tag:
            tagged_sent = model.tag_eval_sentence_crf(sentence, crf)
            for token in tagged_sent:
                y_pred.append(token[1])
        print(classification_report(y_true,
                                    y_pred))

    else:
        print(f"{args.mode} is an incompatible mode. Must be either 'train' or 'tag'.")


if __name__ == '__main__':

    PARSER = argparse.ArgumentParser(description=
                                     """
                                     A basic crf-based POS-tagger.
                                     Accepts either .conllu or tab-delineated
                                     .txt files for training.
                                     """)

    PARSER.add_argument('--mode', metavar='M', type=str, help=
                        """
                        Specifies the tagger mode: {train, tag, eval}.
                        """)
    PARSER.add_argument('--text', metavar='T', type=str, help=
                        """
                        Tags a sentence string.
                        Can only be called if '--mode tag' is specified.
                        """)
    PARSER.add_argument('--config', metavar='C', type=str, help=
                        """
                        A config .yaml file that specifies the train data,
                        model output file, and number of folds for cross-validation.
                        """)

    PARSER.add_argument('--gold', metavar='g', type=str, help=
                        """
                        Specifies the golden standard file.
                        """)
    ARGS = PARSER.parse_args()

    run(ARGS)
