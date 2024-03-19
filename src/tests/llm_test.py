from grag.components.llm import LLM

llm_ = LLM()

models_to_test = ['Llama-2-7b-chat',
                  'Llama-2-13b-chat',
                  'Mixtral-8x7B-Instruct-v0.1',
                  'gemma-7b-it']

pipeline_list = ['llama_cpp', 'hf']


def test_model(model_list, pipeline_list):
    for pipeline in pipeline_list:
        print(f'****** TESTING PIPELINE: {pipeline} ******')
        for model_name in model_list:
            print(f'***** MODEL: {model_name} *****')
            model = llm_.load_model(model_name=model_name,
                                    pipeline=pipeline)
            model.invoke("Who are you?")
            del model, qa_chain


if __name__ == "__main__":
    test_model(models_to_test, pipeline_list)
