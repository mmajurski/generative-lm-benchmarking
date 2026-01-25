import time

from model_interface import SglModel
import prompts
import answer_parser



def build_topic_prompt(context:str):
    prompt = prompts.TOPIC_EXTRACT_PROMPT.format(context=context)
    
    return prompt

def extract_topics_per_context(contexts: list[str], remote:str, model_name:str = None, async_flag:bool=False):

    
    model = SglModel(remote=remote, model=model_name, sync_flag=(not async_flag))

    start_time = time.time()
    prompts = [build_topic_prompt(c) for c in contexts]
    
    results, _ = model.generate(prompts, reasoning_effort="medium")
    total_time = time.time() - start_time
    topic_extraction_responses = []

    total_input_tokens = sum([res['input_tokens'] for res in results])
    total_output_tokens = sum([res['output_tokens'] for res in results])
    total_tokens = sum([res['total_tokens'] for res in results])
    print(f"total input tokens: {total_input_tokens}")
    print(f"total output tokens: {total_output_tokens}")
    print(f"total tokens: {total_tokens}")
    print(f"total toks/sec: {total_tokens / total_time}")
    print(f"total output toks/sec: {total_output_tokens / total_time}")
    
    # Parse the topics from the model response
    topics_list = []
    for i in range(len(contexts)):
        if results[i]['error'] is not None:
            topics_list.append([])
            continue
        
        vals = answer_parser.parse_topic_extraction(results[i]['content'])
        topic_extraction_responses.append(results[i]['content'])
        topics_list.append(vals)

    return topics_list, topic_extraction_responses





if __name__ == '__main__':
   
    start_time = time.time()

    dataset = './source_data/Current Solutions and Future Trends for Robotic Prosthetic Hands.txt'
    with open(dataset, 'r') as f:
        context = f.read()

    topics = extract_topics_per_context([context], 'sierra')
    print(topics)
    