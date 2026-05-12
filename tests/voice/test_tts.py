from pathlib import Path
from unittest.mock import MagicMock

from brain.voice import tts


class DummyPaths:
    PIPER_BIN = Path("/tmp/piper")
    PIPER_MODEL = Path("/tmp/model.onnx")


def test_speak_output_file_uses_arg_list_and_stdin(monkeypatch):
    monkeypatch.setattr(tts, "Paths", DummyPaths)
    monkeypatch.setattr(Path, "exists", lambda self: True)

    run_mock = MagicMock()
    monkeypatch.setattr(tts.subprocess, "run", run_mock)

    special_text = 'Hello $USER; "quoted" `tick`\nnext line'
    cleaned_text = tts._clean_for_speech(special_text)
    tts.speak(special_text, output_file="/tmp/out.wav")

    run_mock.assert_called_once()
    args, kwargs = run_mock.call_args
    assert args[0] == [
        "/tmp/piper",
        "--model",
        "/tmp/model.onnx",
        "--output_file",
        "/tmp/out.wav",
    ]
    assert kwargs["input"] == cleaned_text
    assert kwargs["text"] is True
    assert "shell" not in kwargs


def test_speak_playback_uses_popen_pipe_and_stdin(monkeypatch):
    monkeypatch.setattr(tts, "Paths", DummyPaths)
    monkeypatch.setattr(Path, "exists", lambda self: True)

    piper_proc = MagicMock()
    piper_proc.stdout = MagicMock()
    aplay_proc = MagicMock()

    popen_mock = MagicMock(side_effect=[piper_proc, aplay_proc])
    monkeypatch.setattr(tts.subprocess, "Popen", popen_mock)

    special_text = "semi; $dollar `backtick` \"quote\"\nline2"
    cleaned_text = tts._clean_for_speech(special_text)
    tts.speak(special_text)

    first_call = popen_mock.call_args_list[0]
    second_call = popen_mock.call_args_list[1]

    assert first_call.args[0] == ["/tmp/piper", "--model", "/tmp/model.onnx", "--output_raw"]
    assert first_call.kwargs["stdin"] == tts.subprocess.PIPE
    assert first_call.kwargs["stdout"] == tts.subprocess.PIPE
    assert first_call.kwargs["text"] is True

    assert second_call.args[0] == ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q"]
    assert second_call.kwargs["stdin"] == piper_proc.stdout

    piper_proc.communicate.assert_called_once_with(cleaned_text, timeout=60)
    aplay_proc.wait.assert_called_once_with(timeout=60)


def test_speak_empty_cleaned_text_does_not_invoke_subprocess(monkeypatch):
    monkeypatch.setattr(tts, "Paths", DummyPaths)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(tts, "_clean_for_speech", lambda _text: "   ")

    run_mock = MagicMock()
    popen_mock = MagicMock()
    monkeypatch.setattr(tts.subprocess, "run", run_mock)
    monkeypatch.setattr(tts.subprocess, "Popen", popen_mock)

    tts.speak("ignored")

    run_mock.assert_not_called()
    popen_mock.assert_not_called()
