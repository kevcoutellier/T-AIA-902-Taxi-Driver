from benchmark import run_benchmark


class TestBenchmark:
    def test_benchmark_returns_results_for_all_agents(self):
        results = run_benchmark(train_episodes=10, test_episodes=5, verbose=False)
        assert len(results) == 4
        for name, data in results.items():
            assert "train" in data
            assert "test" in data
            assert "params" in data
