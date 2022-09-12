from jina import Document, DocumentArray, Flow

from executor.spell_checker import SpellChecker


def test_spell_check_integration(incorrect_text, correct_text, input_training_data,
                                 tmpdir):
    train_docs = DocumentArray([Document(content=t) for t in input_training_data])

    with Flow().add(name='spell',
                    uses=SpellChecker,
                    uses_metas={'workspace': str(tmpdir)}) as f:
        f.post(on='/train', inputs=train_docs,
               parameters={
                   'min_freq': 0,
                   'foo': 'break' # this will be ignored
               })

        input_docs = DocumentArray([Document(content=t) for t in incorrect_text])
        results = f.post(on='/index', inputs=input_docs, return_results=True)

        assert len(input_docs) == len(incorrect_text)
        for crafted_doc, expected in zip(results, correct_text):
            assert crafted_doc.content == expected
