from transformers import AutoTokenizer, AutoModel
import torch


# CLS Pooling - Take output from first token
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Encode text
def encode(tokenizer, model, texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return embeddings


def main():
    query = ["test 1", "test 2 3", "test 3 4 5"]

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

    # Encode query and docs
    query_emb = encode(tokenizer, model, query)
    print(query_emb)


if __name__ == '__main__':
    main()
