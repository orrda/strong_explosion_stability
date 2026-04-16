def test_reverse():
    assert reverse_function("hello") == "olleh"
    assert reverse_function("world") == "dlrow"
    assert reverse_function("Python") == "nohtyP"
    assert reverse_function("") == ""