"""文件工具与 Shell 工具回归测试：Glob/Grep 检索、分页读取、唯一替换与原子写入，
以及异步 shell 的固定工作目录、输出截断与超时进程树终止。

全部离线运行：
    .venv/Scripts/python.exe -m unittest mini_agent.tests.test_file_tools -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from mini_agent.agent.tools.builtin import (
    edit_file,
    glob,
    grep,
    read_file,
    run_shell,
    write_file,
)
from mini_agent.agent.tools.policy import configure_policy, workspace_root
from mini_agent.agent.tools.registry import REGISTRY


class FileToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.previous_workspace = workspace_root()
        configure_policy(workspace=self.root)
        self.addCleanup(configure_policy, workspace=self.previous_workspace)

        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text(
            "def handler():\n    return 'ok'\n\n# TODO: 整理\n", encoding="utf-8"
        )
        (self.root / "README.md").write_text("# 项目\nTODO: 写文档\n", encoding="utf-8")
        (self.root / "node_modules" / "dep").mkdir(parents=True)
        (self.root / "node_modules" / "dep" / "index.js").write_text(
            "TODO = '依赖里不该被搜到'\n", encoding="utf-8"
        )

    def test_glob_recursive_and_ignores_dependency_dirs(self) -> None:
        result = glob("**/*.py")
        self.assertIn("src/app.py", result.splitlines())
        self.assertNotIn("node_modules", result)

    def test_glob_root_pattern_stays_at_root(self) -> None:
        self.assertEqual(glob("*.py"), "未找到匹配 '*.py' 的文件")
        self.assertIn("README.md", glob("*.md"))

    def test_glob_cannot_escape_workspace(self) -> None:
        with self.assertRaises(PermissionError):
            glob("*", path="..")

    def test_grep_returns_path_line_and_content(self) -> None:
        result = grep("TODO")
        self.assertIn("README.md:2: TODO: 写文档", result)
        self.assertIn("src/app.py:4: # TODO: 整理", result)
        self.assertNotIn("node_modules", result)

    def test_grep_supports_include_and_case_sensitivity(self) -> None:
        self.assertIn("src/app.py:1:", grep("def", include="*.py"))
        self.assertEqual(grep("todo", case_sensitive=True), "未找到匹配 'todo' 的内容")
        self.assertIn("TODO", grep("todo", case_sensitive=False))

    def test_grep_rejects_invalid_regex(self) -> None:
        with self.assertRaises(ValueError):
            grep("(")

    def test_grep_skips_binary_files(self) -> None:
        (self.root / "blob.bin").write_bytes(b"\x00TODO\x00")
        self.assertNotIn("blob.bin", grep("TODO"))

    def test_read_file_paginates_with_line_numbers(self) -> None:
        result = read_file("src/app.py", start_line=2, max_lines=2)
        self.assertIn("     2\t    return 'ok'", result)
        self.assertIn("     3\t", result)
        self.assertIn("start_line=4", result)

    def test_read_file_can_disable_line_numbers(self) -> None:
        self.assertEqual(read_file("README.md", line_numbers=False), "# 项目\nTODO: 写文档\n")

    def test_edit_file_requires_unique_match(self) -> None:
        write_file("dup.txt", "same\nsame\n")
        with self.assertRaises(ValueError):
            edit_file("dup.txt", "same", "changed")

    def test_edit_file_replaces_all_on_request(self) -> None:
        write_file("dup.txt", "same\nsame\n")
        message = edit_file("dup.txt", "same", "changed", replace_all=True)
        self.assertIn("2 处", message)
        self.assertEqual(read_file("dup.txt", line_numbers=False), "changed\nchanged\n")

    def test_write_file_leaves_no_temp_files(self) -> None:
        write_file("atomic.txt", "内容\n")
        leftovers = [p.name for p in self.root.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])


class ShellToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.previous_workspace = workspace_root()
        configure_policy(workspace=self.root, allow_shell=True)
        self.addCleanup(
            configure_policy, workspace=self.previous_workspace, allow_shell=False
        )

    @staticmethod
    def _python(code: str) -> str:
        return f'"{sys.executable}" -c "{code}"'

    async def test_run_shell_is_registered_as_async(self) -> None:
        self.assertTrue(REGISTRY["run_shell"].is_async)

    async def test_run_shell_uses_workspace_and_truncates_output(self) -> None:
        result = await run_shell(self._python("import os; print(os.getcwd())"))
        self.assertIn("退出码 0", result)
        reported = result.splitlines()[1]
        self.assertEqual(
            os.path.normcase(os.path.realpath(reported)),
            os.path.normcase(os.path.realpath(self.root)),
        )

        long_result = await run_shell(self._python("print('x' * 6000)"))
        self.assertIn("输出过长，已截断", long_result)
        self.assertLess(len(long_result), 6000)

    async def test_timeout_terminates_process_tree(self) -> None:
        (self.root / "late.py").write_text(
            "import time\n"
            "time.sleep(2)\n"
            "open('late.txt', 'w', encoding='utf-8').write('x')\n",
            encoding="utf-8",
        )
        (self.root / "spawner.py").write_text(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, 'late.py'])\n"
            "time.sleep(10)\n",
            encoding="utf-8",
        )

        started = time.perf_counter()
        result = await run_shell(f'"{sys.executable}" spawner.py', timeout=1)
        elapsed = time.perf_counter() - started

        self.assertIn("已终止进程树", result)
        self.assertLess(elapsed, 5)
        await asyncio.sleep(2.5)
        self.assertFalse(
            (self.root / "late.txt").exists(), "超时后子进程仍写入了文件"
        )

    async def test_run_shell_requires_allow_shell(self) -> None:
        configure_policy(allow_shell=False)
        with self.assertRaises(PermissionError):
            await run_shell("echo hi")


if __name__ == "__main__":
    unittest.main()
