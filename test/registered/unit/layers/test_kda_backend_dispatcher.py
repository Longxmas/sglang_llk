import unittest
from unittest.mock import Mock

from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKDAKernelDispatcher(unittest.TestCase):
    def test_target_verify_forwards_backend_specific_kwargs(self):
        dispatcher = KDAKernelDispatcher.__new__(KDAKernelDispatcher)
        dispatcher.verify_kernel = Mock()
        expected_output = object()
        dispatcher.verify_kernel.target_verify.return_value = expected_output

        arguments = {
            name: object()
            for name in (
                "A_log",
                "dt_bias",
                "q",
                "k",
                "v",
                "a",
                "b",
                "ssm_states",
                "cache_indices",
                "query_start_loc",
                "intermediate_states_buffer",
                "intermediate_state_indices",
                "retrieve_parent_token",
            )
        }
        replay_scratch = {
            "replayssm_d": object(),
            "replayssm_k": object(),
            "replayssm_g": object(),
        }

        output = dispatcher.target_verify(
            **arguments,
            cache_steps=4,
            lower_bound=object(),
            **replay_scratch,
        )

        self.assertIs(output, expected_output)
        forwarded = dispatcher.verify_kernel.target_verify.call_args.kwargs
        for name, value in replay_scratch.items():
            self.assertIs(forwarded[name], value)
        self.assertIn("lower_bound", forwarded)


if __name__ == "__main__":
    unittest.main()
