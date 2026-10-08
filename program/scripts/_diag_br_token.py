"""One-off: does chexagent's tokenizer emit literal `<br>` from a special token?"""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/CheXagent-8b-fp16", trust_remote_code=True)
print("tokenizer class:", type(tok).__name__)
print("additional_special_tokens:", tok.additional_special_tokens)
print("all_special_tokens:", tok.all_special_tokens)
print("decode([br_id]):", tok.decode([tok.convert_tokens_to_ids("<br>")]) if "<br>" in tok.get_vocab() else "not a single token")

encode_br = tok.encode("<br>")
print("encode('<br>'):", encode_br)
print("encode('a<br>b'):", tok.encode("a<br>b"))
print("encode('a\nb'):", tok.encode("a\nb"))

# where does a raw <br> show up -- as one token or split?
ids = tok.encode("x<br>y")
print("decode('x<br>y' -> ids):", ids, "->", tok.decode(ids))