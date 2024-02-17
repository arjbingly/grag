import textwrap


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    # print(wrap_text_preserve_newlines(llm_response['result']))
    print(llm_response['result'])
    print('\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
