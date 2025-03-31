from fastapi import FastAPI, Query
from typing import List
from torch import load, tensor, cuda, long, no_grad, topk
import transformer_utils
from sentencepiece import SentencePieceProcessor



# Load SentencePiece tokenizer
tokenizer = SentencePieceProcessor()
tok_path  = r"C:\Users\Ans\Desktop\code\09_NLP_Project\Pre-Training__and__Full_Fine-Tuning__of__a__Transformer\05_Deployment\sentencepiece.model"
tokenizer.load(tok_path)

# Define model parameters
num_layers                 = 2
embedding_dim              = 128
fully_connected_dim        = 128
num_heads                  = 2
positional_encoding_length = 256

encoder_vocab_size = int(tokenizer.vocab_size())
decoder_vocab_size = encoder_vocab_size

# Initialize and load model
transformer = transformer_utils.Transformer(
    num_layers,
    embedding_dim,
    num_heads,
    fully_connected_dim,
    encoder_vocab_size,
    decoder_vocab_size,
    positional_encoding_length,
    positional_encoding_length,
)

device_ = 'cuda' if cuda.is_available() else 'cpu'
path = r"C:\Users\Ans\Desktop\code\09_NLP_Project\Pre-Training__and__Full_Fine-Tuning__of__a__Transformer\05_Deployment\best_qA_model_106th_epoch.pt"

checkpoint = load(path, map_location=device_, weights_only=True)
transformer.load_state_dict(checkpoint['model_state_dict'])
transformer.to(device_)
transformer.eval()

app = FastAPI()

@app.get("/predict", response_model=List[str])
def predict(example_question: str = Query(..., description="The input question prompt")):

    eval_inp     = tokenizer.tokenize(example_question)
    eval_tar_inp = tokenizer.tokenize('answer: ')

    eval_inp     = tensor(eval_inp,     dtype=long, device=device_).unsqueeze(0)
    eval_tar_inp = tensor(eval_tar_inp, dtype=long, device=device_).unsqueeze(0)

    with no_grad():
        eval_preds, _    =  transformer(eval_inp, eval_tar_inp)
        _, topk_indices  =  topk(eval_preds[:, -1, :], k=10, dim=-1)
        topk_indices     =  topk_indices.int().tolist()
        decoded_answers  =  [tokenizer.detokenize([idx]) for idx in topk_indices[0]]

    return decoded_answers
