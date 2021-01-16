import waitress
from flask import Flask, render_template
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, render_template


app = Flask(__name__)

model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model.load_state_dict(torch.load('models/model1'))
model.eval()


@app.route('/')
def home():
	return render_template('home.html')
 

@app.route('/invocations')
def generateTweet(prompt):
	with torch.no_grad():
		output_list = []

		for tweet_idx in range(10):
			cur_ids = torch.tensor(tokenizer.encode(f"@realDonaldTrump: {prompt}")).unsqueeze(0).to(device)

			finished = False
			counter = 0

			while finished == False:
			    outputs = model(cur_ids, labels=cur_ids)
			    loss, logits = outputs[:2]
			    softmax_logits = torch.softmax(logits[0,-1], dim=0)

			    if counter < 3:
			      n = 100
			    else:
			      n = 3

			    next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) 
			    cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) 

			    if next_token_id in tokenizer.encode('<EOS>') or counter > 100:
			      tweet = tokenizer.decode(list(cur_ids.squeeze().to('cpu').numpy()))
		    
			      tweet = re.sub('([^A-Za-z@:\.])+', ' ', tweet)

			      output_list.append(tweet)
			      finished = True
			
				
			counter += 1
		
	
	return render_template('home.html', tweets=output_list)

if __name__=='__main__':
	waitress.serve(app, port=5000)
