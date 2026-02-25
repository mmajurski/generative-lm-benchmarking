import os
import httpx
import openai
import asyncio
import time
import prompts

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def translate_remote(remote:str) -> tuple[str, str]:
    if ":" in remote:
        remote, port_num = remote.split(":")
        port_num = int(port_num)
    else:
        port_num = 8443
    if remote == "openai":
        url = "https://api.openai.com/v1"
        api_key = os.environ.get("OPENAI_API_KEY")
    elif remote == "rchat":
        url = "https://rchat.nist.gov/api"
        api_key = os.environ.get("RCHAT_API_KEY")
    else:
        url=f"https://{remote}.nist.gov:{port_num}/v1"
        api_key = "VLLM_API_KEY"

        
    
    if api_key is None:
        raise ValueError("API key is not set for remote: %s" % remote)
    if url is None:
        raise ValueError("URL is not set for remote: %s" % remote)
    
    return url, api_key




class SglModel:
    def __init__(self, model=None, remote=None, sync_flag: bool = False):
        if model is None:
            raise RuntimeError("Model is not set")
        if remote is None:
            raise RuntimeError("Remote is not set")
        self.model = model
        self.remote = remote
        self.sync_flag = sync_flag
        self.url, self.api_key = translate_remote(remote)

        self.connection_parallelism = 64
        if 'openai' in self.url:
            print("OpenAI detected, setting connection parallelism to 8")
            self.connection_parallelism = 8
        if 'rchat' in self.url:
            print("RCHAT detected, setting connection parallelism to 32")
            self.connection_parallelism = 32

        print(f"Using model: {self.model} on remote: {self.remote} at {self.url} (sync={sync_flag})")

        # Create appropriate client based on sync_flag
        if self.sync_flag:
            self.client = openai.OpenAI(
                base_url=self.url,
                api_key=self.api_key,
                http_client=httpx.Client(verify=False)
            )
        else:
            self.client = openai.AsyncOpenAI(
                base_url=self.url,
                api_key=self.api_key,
                http_client=httpx.AsyncClient(verify=False)
            )


    def _generate_text_sync(self, prompt, request_id, reasoning_effort="high"):
        """Synchronous text generation for debugging"""
        start_time = time.time()
        try:
            if 'openai' in str(self.client.base_url):
                # Use OpenAI Responses API for OpenAI models
                response = self.client.responses.create(
                    model=self.model,
                    temperature=0.7,
                    stream=False,
                    max_output_tokens=16000,
                    # reasoning={"effort": reasoning_effort},
                    input=[{"role": "user", "content": prompt}]
                )
                elapsed = time.time() - start_time
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_tokens = input_tokens + output_tokens
                content = response.output_text

                result = {
                    "request_id": request_id,
                    "content": content,
                    "error": None,
                    "elapsed_time": elapsed,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
                if content is None or content.isspace():
                    result['error'] = "Empty response"
                return result
            else:
                # Use Chat Completions API for other models
                messages = [
                    {"role": "system", "content": prompts.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=32000,
                    temperature=0.7,
                    reasoning_effort=reasoning_effort,
                    stream=False
                )
                elapsed = time.time() - start_time
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                result = {
                    "request_id": request_id,
                    "content": response.choices[0].message.content,
                    "error": None,
                    "elapsed_time": elapsed,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
                if response.choices[0].message.content.isspace():
                    result['error'] = "Empty response"
                return result
        except Exception as e:
            result = {
                "request_id": request_id,
                "content": None,
                "error": str(e),
                "elapsed_time": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
            return result

    @staticmethod
    async def _generate_text_async(model, client, prompt, request_id, reasoning_effort="high"):
        """Asynchronous text generation for parallel processing"""
        start_time = time.time()
        try:
            if 'openai' in str(client.base_url):
                # Use OpenAI Responses API for OpenAI models
                response = await client.responses.create(
                    model=model,
                    temperature=0.7,
                    stream=False,
                    max_output_tokens=16000,
                    # reasoning={"effort": reasoning_effort},
                    input=[{"role": "user", "content": prompt}]
                )
                elapsed = time.time() - start_time
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_tokens = input_tokens + output_tokens
                content = response.output_text

                result = {
                    "request_id": request_id,
                    "content": content,
                    "error": None,
                    "elapsed_time": elapsed,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
                if content is None or content.isspace():
                    result['error'] = "Empty response"
                return result
            else:
                # Use Chat Completions API for other models
                messages = [
                    {"role": "system", "content": prompts.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=32000,
                    temperature=0.7,
                    reasoning_effort=reasoning_effort,
                    stream=False
                )
                elapsed = time.time() - start_time
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                result = {
                    "request_id": request_id,
                    "content": response.choices[0].message.content,
                    "error": None,
                    "elapsed_time": elapsed,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
                if response.choices[0].message.content.isspace():
                    result['error'] = "Empty response"
                return result
        except Exception as e:
            result = {
                "request_id": request_id,
                "content": None,
                "error": str(e),
                "elapsed_time": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
            return result
        

    def _process_all_sync(self, model_prompts, reasoning_effort="high"):
        """Process all prompts sequentially (for debugging)"""
        total_start_time = time.time()
        results = []

        for i, prompt in enumerate(model_prompts):
            current_time = time.strftime("%H:%M:%S")
            print(f"  ({current_time}) sync request {i + 1}/{len(model_prompts)}")
            result = self._generate_text_sync(prompt, i, reasoning_effort)
            results.append(result)

        total_time = time.time() - total_start_time
        return results, total_time

    async def _process_batch_async(self, batch_prompts, start_idx, reasoning_effort):
        """Process a single batch of prompts concurrently"""
        batch_tasks = [
            self._generate_text_async(self.model, self.client, prompt, start_idx + j, reasoning_effort)
            for j, prompt in enumerate(batch_prompts)
        ]
        return await asyncio.gather(*batch_tasks)

    async def _process_all_async(self, model_prompts, reasoning_effort="high"):
        """Process all batches with a single client instance, in chunks"""
        total_start_time = time.time()
        results = []

        batch_size = self.connection_parallelism
        for i in range(0, len(model_prompts), batch_size):
            batch_prompts = model_prompts[i:i + batch_size]
            current_time = time.strftime("%H:%M:%S")
            print(f"  ({current_time}) async batch {i // batch_size + 1}/{(len(model_prompts) - 1) // batch_size + 1} ({len(batch_prompts)} prompts per batch)")

            batch_results = await self._process_batch_async(batch_prompts, i, reasoning_effort)
            results.extend(batch_results)

        total_time = time.time() - total_start_time
        return results, total_time

    def generate(self, model_prompts: list[str], reasoning_effort="high"):
        mode = "synchronously" if self.sync_flag else "asynchronously"
        print(f"Remote model generating {len(model_prompts)} prompts ({mode}) using reasoning effort: {reasoning_effort}")

        if self.sync_flag:
            return self._process_all_sync(model_prompts, reasoning_effort)
        else:
            return asyncio.run(self._process_all_async(model_prompts, reasoning_effort))
    

    


# Example usage
if __name__ == "__main__":
    import json

    # Async mode (default, parallel processing)
    model = SglModel(sync_flag=False)

    # Sync mode (for debugging, sequential processing)
    # model = SglModel(sync_flag=True)

    # Load the squadv2 dataset subset
    with open('squadv2_subset.json', 'r') as f:
        dataset = json.load(f)
    dataset = dataset[:32]

    contexts = [item['context'] for item in dataset]

    results, total_time = model.generate(contexts)

    res = results[0]
    print(res)
    print(f"in total took: {total_time} seconds")
    print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")
    
    