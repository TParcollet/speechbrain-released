#!/usr/bin/env/python3
"""Recipe for training a sequence-to-sequence ST system with Fisher-Callhome.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beam search coupled with a neural
language model.

To run this recipe, do the following:
> python train.py hparams/train_bpe_10k.yaml

Authors
 * YAO-FEI, CHENG 2021
"""

import sys
import torch
import logging
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)

        wavs, wav_lens = batch.sig

        # for translation task
        tokens_bos, _ = batch.tokens_bos

        # for asr task
        # transcription_bos, _ = batch.transcription_bos

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        # forward modules
        src = self.hparams.CNN(feats)
        enc_out, pred = self.hparams.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        asr_p_seq = None
        # asr output layer for seq2seq log-probabilities
        # if self.hparams.asr_weight > 0 and self.hparams.ctc_weight < 1:
        #    asr_pred = self.hparams.Transformer.forward_asr(
        #        enc_out, transcription_bos, pad_idx=self.hparams.pad_index,
        #    )
        #    asr_pred = self.hparams.asr_seq_lin(asr_pred)
        #    asr_p_seq = self.hparams.log_softmax(asr_pred)

        # st output layer for seq2seq log-probabilities
        pred = self.hparams.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # asr ctc
        p_ctc = None
        if self.hparams.ctc_weight > 0:
            logits = self.hparams.ctc_lin(enc_out)
            p_ctc = self.hparams.log_softmax(logits)

        # mt task
        mt_p_seq = None
        # if self.hparams.mt_weight > 0:
        #    _, mt_pred = self.hparams.Transformer.forward_mt(
        #        transcription_bos, tokens_bos, pad_idx=self.hparams.pad_index,
        #    )

        #    # mt output layer for seq2seq log-probabilities
        #    mt_pred = self.hparams.seq_lin(mt_pred)
        #    mt_p_seq = self.hparams.log_softmax(mt_pred)

        # compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, asr_p_seq, mt_p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        (p_ctc, p_seq, asr_p_seq, mt_p_seq, wav_lens, hyps,) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        # loss for different task
        attention_loss = 0
        asr_ctc_loss = 0
        asr_attention_loss = 0
        mt_loss = 0

        # st attention loss
        attention_loss = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens,
        )

        # asr attention loss
        # if self.hparams.ctc_weight < 1 and self.hparams.asr_weight > 0:
        #    asr_attention_loss = self.hparams.seq_cost(
        #        asr_p_seq, tokens_eos, length=transcription_eos_lens,
        #    )

        # asr ctc loss
        # if self.hparams.ctc_weight > 0 and self.hparams.asr_weight > 0:
        #    asr_ctc_loss = self.hparams.ctc_cost(
        #        p_ctc, transcription_tokens, wav_lens, transcription_lens,
        #    )

        # mt attention loss
        if self.hparams.mt_weight > 0:
            mt_loss = self.hparams.seq_cost(
                mt_p_seq, tokens_eos, length=tokens_lens,
            )

        asr_loss = (self.hparams.ctc_weight * asr_ctc_loss) + (
            1 - self.hparams.ctc_weight
        ) * asr_attention_loss
        loss = (
            (1 - self.hparams.asr_weight - self.hparams.mt_weight)
            * attention_loss
            + self.hparams.asr_weight * asr_loss
            + self.hparams.mt_weight * mt_loss
        )

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = self.tokenizer(hyps, task="decode_from_list")

            # Convert indices to words
            target_words = undo_padding(tokens_eos, tokens_eos_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
            self.bleu_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        self.check_and_reset_optimizer()
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # anneal lr every update
            self.hparams.noam_annealing(self.optimizer)

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.bleu_metric = self.hparams.bleu_computer()
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            if stage == sb.Stage.TEST:
                stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            elif (
                current_epoch % valid_search_interval == 0
                and stage == sb.Stage.VALID
            ):
                stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            current_epoch = self.hparams.epoch_counter.current

            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.stage_one_epochs:
                lr = self.hparams.noam_annealing.current_lr
                steps = self.hparams.noam_annealing.n_steps
                optimizer = self.optimizer.__class__.__name__
            else:
                lr = self.hparams.lr_sgd
                steps = -1
                optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

            with open(self.hparams.bleu_file, "a+", encoding="utf-8") as w:
                self.bleu_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def check_and_reset_optimizer(self):
        """reset the optimizer if training enters stage 2"""
        current_epoch = self.hparams.epoch_counter.current
        if not hasattr(self, "switched"):
            self.switched = False
            if isinstance(self.optimizer, torch.optim.SGD):
                self.switched = True

        if self.switched is True:
            return

        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            self.switched = True

    def on_fit_start(self):
        """Initialize the right optimizer on the training start"""
        super().on_fit_start()

        # if the model is resumed from stage two, reinitialize the optimizer
        current_epoch = self.hparams.epoch_counter.current
        current_optimizer = self.optimizer
        if current_epoch > self.hparams.stage_one_epochs:
            del self.optimizer
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            # Load latest checkpoint to resume training if interrupted
            if self.checkpointer is not None:

                # do not reload the weights if training is interrupted right before stage 2
                group = current_optimizer.param_groups[0]
                if "momentum" not in group:
                    return

                self.checkpointer.recover_if_possible(
                    device=torch.device(self.device)
                )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()


# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_json"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        train_data = train_data.filtered_sorted(
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_json"],
        replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_he_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_he_json"],
        replacements={"data_root": data_folder},
    )

    test_com_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_com_json"],
        replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_he_data = test_he_data.filtered_sorted(sort_key="duration")
    test_com_data = test_com_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_he_data, test_com_data]

    # defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_json"],
        annotation_read="wrd_tgt",
        annotation_format="json",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        bos_id=hparams["bos_index"],
        eos_id=hparams["eos_index"],
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "duration", "offset")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, duration, offset):
        info = torchaudio.info(wav)
        start = int(offset * 16000)
        stop = int((offset + duration) * 16000)
        speech_segment = {"file": wav, "start": start, "stop": stop}
        sig = sb.dataio.dataio.read_audio(speech_segment)
        if info.num_channels > 1:
            sig = torch.mean(sig, dim=1)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd_tgt")
    @sb.utils.data_pipeline.provides(
        "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd_tgt):
        tokens_list = tokenizer.sp.encode_as_ids(wrd_tgt)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_he_data, test_com_data, tokenizer


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset preparation (parsing CommonVoice)
    from mustc_v1_prepare import prepare_mustc_v1  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_mustc_v1,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "font_case": hparams["font_case"],
            "accented_letters": hparams["accented_letters"],
            "punctuation": hparams["punctuation"],
            "non_verbal": hparams["non_verbal"],
            "tgt_language": hparams["tgt_language"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create the datasets objects as well as tokenization and encoding :-D
    (
        train_data,
        valid_data,
        test_he_data,
        test_com_datam,
        tokenizer,
    ) = dataio_prepare(hparams)

    st_brain = ST(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    st_brain.fit(
        st_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    st_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test.txt"
    st_brain.evaluate(
        test_he_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
