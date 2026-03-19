from types import SimpleNamespace

from chronicals.training.chronicals_trainer import ChronicalsTrainer


def make_trainer(log_progress_pct: bool):
    trainer = ChronicalsTrainer.__new__(ChronicalsTrainer)
    trainer.scheduler = None
    trainer.optimizer = SimpleNamespace(param_groups=[{"lr": 2e-4}])
    trainer.args = SimpleNamespace(
        max_steps=100,
        log_progress_pct=log_progress_pct,
        log_progress_flush=False,
        report_every_n_steps=100,
    )
    trainer.state = SimpleNamespace(global_step=25, log_history=[], epoch=0.0)
    trainer.reporter = None
    trainer.track_mfu = False
    return trainer


def test_progress_logging_includes_percentage(capsys):
    trainer = make_trainer(log_progress_pct=True)
    trainer._log_step_fast(loss=1.2345, tokens_per_sec=999.0)
    out = capsys.readouterr().out
    assert "25/100" in out
    assert "(25.0%)" in out


def test_progress_logging_can_hide_percentage(capsys):
    trainer = make_trainer(log_progress_pct=False)
    trainer._log_step_fast(loss=1.2345, tokens_per_sec=999.0)
    out = capsys.readouterr().out
    assert "Step 25:" in out
    assert "(25.0%)" not in out
