from necast.simulator.assets import asset_path
from necast.simulator.config import Config
from necast.simulator.ne_templates import load_empirical_ne_templates
from necast.simulator.stdpopsim_templates import load_stdpopsim_templates


def test_shipped_data_is_found_from_any_directory(tmp_path, monkeypatch):
    # a missing file would not fail loudly: the simulator would quietly draw other histories
    monkeypatch.chdir(tmp_path)
    config = Config()
    assert asset_path(config.recomb_map).exists() and asset_path(config.mut_map).exists()
    store = load_empirical_ne_templates(config)
    assert len(store.templates) == 191
    assert store.source_counts == {"smcpp_1kg": 26, "phlash_unified": 154, "phlash_super": 7,
                                   "msmc2_example": 2, "msmc_im_example": 2}
    assert len(load_stdpopsim_templates(config)) == 18
