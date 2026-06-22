from orbit.backends.sglang_utils.native_ops import force_native_forward_after_init


def test_force_native_forward_after_init_uses_native_forward():
    class Op:
        def __init__(self):
            self._forward_method = self.forward_cuda

        def forward_cuda(self):
            return "cuda"

        def forward_native(self):
            return "native"

    force_native_forward_after_init(Op)

    assert Op()._forward_method() == "native"


def test_force_native_forward_after_init_is_idempotent():
    class Op:
        init_count = 0

        def __init__(self):
            type(self).init_count += 1
            self._forward_method = self.forward_cuda

        def forward_cuda(self):
            return "cuda"

        def forward_native(self):
            return "native"

    force_native_forward_after_init(Op)
    force_native_forward_after_init(Op)
    instance = Op()

    assert instance._forward_method() == "native"
    assert Op.init_count == 1
