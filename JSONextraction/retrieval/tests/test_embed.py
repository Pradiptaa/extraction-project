"""The real `retrieval.embed.Embedder`, against an Ollama server faked at the
HTTP boundary — every other test replaces the Embedder wholesale.

The invariants here outlived the move from a hosted API to a local one: a
mis-shaped batch corrupts the collection whoever served it.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import unittest
from unittest import mock

import httpx
from tenacity import wait_none

from retrieval.embed import Embedder, ModelNotAvailable, is_retryable

URL = "http://localhost:11434/api/embed"


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", URL)
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


def _response(vectors: list[list[float]], tokens: int = 7, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={"embeddings": vectors, "prompt_eval_count": tokens},
        request=httpx.Request("POST", URL),
    )


class RetryPolicyTests(unittest.TestCase):
    def test_server_faults_are_retried(self) -> None:
        for status in (429, 500, 502, 503, 504):
            with self.subTest(status=status):
                self.assertTrue(is_retryable(_http_error(status)))

    def test_client_errors_fail_immediately(self) -> None:
        """Retrying an unpulled model hides the real message behind a minute of backoff."""
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.assertFalse(is_retryable(_http_error(status)))

    def test_a_server_that_is_down_or_restarting_is_waited_out(self) -> None:
        """Ollama is a local process; `ask` may well race its startup."""
        request = httpx.Request("POST", URL)
        self.assertTrue(is_retryable(httpx.ConnectError("refused", request=request)))
        self.assertTrue(is_retryable(httpx.ReadTimeout("slow", request=request)))
        self.assertTrue(is_retryable(TimeoutError()))
        self.assertTrue(is_retryable(ConnectionError()))

    def test_programming_errors_are_not_retried(self) -> None:
        self.assertFalse(is_retryable(ValueError("bug")))
        self.assertFalse(is_retryable(RuntimeError("dimension changed")))
        self.assertFalse(is_retryable(ModelNotAvailable("not pulled")))


class EmbedderTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch("retrieval.embed.httpx.Client")
        self.client = patcher.start().return_value
        self.addCleanup(patcher.stop)
        self.embedder = Embedder(model="bge-m3")
        # The retry object is shared by every instance, so restore it rather
        # than leak zero-wait retries into other tests.
        retrying = Embedder._call.retry
        original_wait = retrying.wait
        retrying.wait = wait_none()
        self.addCleanup(setattr, retrying, "wait", original_wait)

    def test_vectors_and_token_usage_are_returned_and_accumulated(self) -> None:
        self.client.post.return_value = _response([[0.1, 0.2], [0.3, 0.4]], tokens=5)
        self.assertEqual(self.embedder.embed(["a", "b"]), [[0.1, 0.2], [0.3, 0.4]])
        self.embedder.embed(["a", "b"])
        self.assertEqual(self.embedder.total_tokens, 10)
        self.assertEqual(self.embedder.dimension, 2)

    def test_the_request_names_the_model_and_sends_the_batch_whole(self) -> None:
        self.client.post.return_value = _response([[0.1], [0.2]])
        self.embedder.embed(["a", "b"])
        url, kwargs = self.client.post.call_args.args[0], self.client.post.call_args.kwargs
        self.assertEqual(url, URL)
        self.assertEqual(kwargs["json"]["model"], "bge-m3")
        self.assertEqual(kwargs["json"]["input"], ["a", "b"])

    def test_truncation_is_refused_rather_than_silent(self) -> None:
        """A truncated input yields a vector for half a clause, and reports success."""
        self.client.post.return_value = _response([[0.1]])
        self.embedder.embed(["a"])
        self.assertIs(self.client.post.call_args.kwargs["json"]["truncate"], False)

    def test_a_host_is_used_as_given_without_a_double_slash(self) -> None:
        self.client.post.return_value = _response([[0.1]])
        Embedder(model="bge-m3", host="http://box:11434/").embed(["a"])
        self.assertEqual(self.client.post.call_args.args[0], "http://box:11434/api/embed")

    def test_a_server_fault_is_retried_until_it_succeeds(self) -> None:
        self.client.post.side_effect = [_response([], status=503), _response([], status=500),
                                        _response([[1.0, 0.0]])]
        self.assertEqual(self.embedder.embed(["a"]), [[1.0, 0.0]])
        self.assertEqual(self.client.post.call_count, 3)

    def test_an_unpulled_model_names_the_command_that_fixes_it(self) -> None:
        self.client.post.return_value = _response([], status=404)
        with self.assertRaises(ModelNotAvailable) as caught:
            self.embedder.embed(["a"])
        self.assertIn("ollama pull bge-m3", str(caught.exception))
        self.assertEqual(self.client.post.call_count, 1, "a missing model is not transient")

    def test_retries_are_bounded(self) -> None:
        self.client.post.return_value = _response([], status=503)
        with self.assertRaises(httpx.HTTPStatusError):
            self.embedder.embed(["a"])
        self.assertEqual(self.client.post.call_count, 6)

    def test_a_dimension_change_mid_run_is_refused(self) -> None:
        """The model changing underneath a job that has already written vectors."""
        self.client.post.side_effect = [_response([[0.1, 0.2]]), _response([[0.1, 0.2, 0.3]])]
        self.embedder.embed(["a"])
        with self.assertRaises(RuntimeError) as caught:
            self.embedder.embed(["b"])
        self.assertIn("dimension changed mid-run", str(caught.exception))

    def test_a_short_response_is_refused_rather_than_misaligned(self) -> None:
        """N texts and N-1 vectors would silently misalign every later row."""
        self.client.post.return_value = _response([[0.1, 0.2]])
        with self.assertRaises(RuntimeError) as caught:
            self.embedder.embed(["a", "b"])
        self.assertIn("refusing to guess", str(caught.exception))

    def test_a_missing_token_count_is_not_an_error(self) -> None:
        """Only ever logged, and older Ollama builds omit it."""
        self.client.post.return_value = httpx.Response(
            200, json={"embeddings": [[0.1]]}, request=httpx.Request("POST", URL)
        )
        self.assertEqual(self.embedder.embed(["a"]), [[0.1]])
        self.assertEqual(self.embedder.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
