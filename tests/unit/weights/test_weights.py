"""Weight loading tests."""

class TestLoadCanvit:
    def test_loads_and_has_correct_config(self, mlx_model):
        cfg = mlx_model.cfg
        assert cfg.embed_dim == 768
        assert cfg.num_heads == 12
        assert cfg.n_blocks == 12
        assert cfg.canvas_num_heads == 8
        assert cfg.canvas_head_dim == 128
        assert cfg.enable_vpe is True

    def test_block_count(self, mlx_model):
        assert len(mlx_model.blocks) == mlx_model.cfg.n_blocks

    def test_rw_schedule(self, mlx_model):
        assert mlx_model.read_after_blocks == [1, 5, 9]
        assert mlx_model.write_after_blocks == [3, 7, 11]
