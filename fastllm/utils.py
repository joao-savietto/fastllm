import re


def strip_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def extract_code_blocks(text):

    code_blocks = re.findall(r'\`\`\`[\s\S]*?\`\`\`', text, flags=re.DOTALL)

    extracted_blocks = []

    for block in code_blocks:
        content = block.strip()
        content = content[3:-3]
        content_split = content.split('\n')
        language = content_split[0]
        code = '\n'.join(content_split[1:])

        json_block = {
            'code': code,
            'language': language,
        }

        extracted_blocks.append(json_block)

    return extracted_blocks
