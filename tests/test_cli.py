"""Tests for CLI entry point."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_fusion.cli import build_parser, main


class TestCLI:
    def test_parser_creation(self):
        parser = build_parser()
        assert parser is not None

    def test_parse_query_command(self):
        parser = build_parser()
        args = parser.parse_args(["query", "What is 2+2?"])
        assert args.command == "query"
        assert args.question == "What is 2+2?"

    def test_parse_query_with_strategy(self):
        parser = build_parser()
        args = parser.parse_args(["query", "Test", "--strategy", "handoff"])
        assert args.strategy == "handoff"

    def test_parse_config_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--config", "/path/to/config.toml", "query", "Test"])
        assert str(args.config) == "/path/to/config.toml"

    def test_parse_health_command(self):
        parser = build_parser()
        args = parser.parse_args(["health"])
        assert args.command == "health"

    def test_parse_cache_cleanup_command(self):
        parser = build_parser()
        args = parser.parse_args(["cache", "cleanup", "--hours", "48"])
        assert args.command == "cache"
        assert args.cache_action == "cleanup"
        assert args.hours == 48

    def test_parse_strategies_command(self):
        parser = build_parser()
        args = parser.parse_args(["strategies"])
        assert args.command == "strategies"

    @pytest.mark.asyncio
    async def test_main_query_success(self, capsys):
        with patch('hermes_fusion.cli.create_engine_from_config') as mock_create:
            mock_engine = AsyncMock()
            mock_result = MagicMock()
            mock_result.final_answer = "Test answer"
            mock_result.confidence = 0.9
            mock_result.method = "weighted_vote"
            mock_result.participating_providers = ["localai", "xai"]
            mock_result.metadata = {}
            mock_engine.query = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_engine

            exit_code = await main(["query", "What is 2+2?"])
            
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "Test answer" in captured.out
            assert "weighted_vote" in captured.out

    @pytest.mark.asyncio
    async def test_main_health_check(self, capsys):
        with patch('hermes_fusion.cli.create_engine_from_config') as mock_create:
            mock_engine = AsyncMock()
            mock_engine.health_check = AsyncMock(return_value={
                "localai": True,
                "xai": False,
                "qdrant": True,
            })
            mock_create.return_value = mock_engine

            exit_code = await main(["health"])
            
            # Exit code 1 because xai is unhealthy
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "localai" in captured.out
            assert "✓" in captured.out or "True" in captured.out

    @pytest.mark.asyncio
    async def test_main_cache_cleanup(self, capsys):
        with patch('hermes_fusion.cli.create_engine_from_config') as mock_create:
            mock_engine = AsyncMock()
            mock_engine.cleanup_cache = AsyncMock(return_value=42)
            mock_create.return_value = mock_engine

            exit_code = await main(["cache", "cleanup", "--hours", "24"])
            
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "42" in captured.out

    @pytest.mark.asyncio
    async def test_main_list_strategies(self, capsys):
        with patch('hermes_fusion.cli.create_engine_from_config') as mock_create:
            mock_engine = AsyncMock()
            mock_engine.get_available_strategies = MagicMock(return_value=[
                "weighted_vote", "handoff", "cot_consensus", "best_of_n"
            ])
            mock_create.return_value = mock_engine

            exit_code = await main(["strategies"])
            
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "weighted_vote" in captured.out
            assert "handoff" in captured.out