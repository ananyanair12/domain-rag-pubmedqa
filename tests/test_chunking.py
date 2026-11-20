from src.chunking import SIZES, chunk_words


def test_sizes_lists_supported_chunk_sizes():
    assert SIZES == {"128": 128, "256": 256, "512": 512, "whole": None}


def test_short_text_produces_one_chunk():
    text = "one two three"

    assert chunk_words(text, 128) == [text]


def test_none_size_preserves_the_whole_text():
    text = "one  two\nthree"

    assert chunk_words(text, None) == [text]


def test_300_words_are_covered_by_overlapping_128_word_chunks():
    words = [f"word{i}" for i in range(300)]
    chunks = chunk_words(" ".join(words), 128)
    chunk_words_list = [chunk.split() for chunk in chunks]

    assert len(chunks) > 1
    assert all(len(chunk) <= 128 for chunk in chunk_words_list)
    assert all(
        set(left).intersection(right)
        for left, right in zip(chunk_words_list, chunk_words_list[1:])
    )
    assert {word for chunk in chunk_words_list for word in chunk} == set(words)
