import traceback

class ExecutionAgent:
    def __init__(self, param_func_map, processor_map, llm_caller, test_cases=None):
        self.PARAM_FUNC_MAP = param_func_map
        self.PROCESSOR_MAP = processor_map
        self.llm_caller = llm_caller
        self.test_cases = test_cases or {}
        self.generated_code = {}   # 保存LLM生成的代码
        self.execution_results = {} # 每个key的执行结果

    def generate_and_inject_code(self, key, kind='param'):
        prompt = self._build_prompt(key, kind)
        code = self.llm_caller(prompt)
        self.generated_code[key] = code
        local_env = {}
        try:
            exec(code, {}, local_env)
            # 假定函数名和key一致
            func = local_env.get(key)
            if func:
                if kind == 'param':
                    self.PARAM_FUNC_MAP[key] = func
                else:
                    self.PROCESSOR_MAP[key] = func
            else:
                print(f"Warning: 函数{key}未在代码中正确定义")
        except Exception as e:
            print(f"代码执行失败: {key}\n{code}\n{traceback.format_exc()}")

    def _build_prompt(self, key, kind='param'):

        if kind == 'param':
            return f"请为以下参数生成一个Python函数，函数名为 {key}，输入参数可自定义，返回值尽量模拟实际业务。"
        else:
            return f"请为以下后处理器生成一个Python函数，函数名为 {key}，输入参数和返回值可自定义，尽量合理模拟实际业务。"

    def verify_and_run(self, key, kind='param'):

        func = self.PARAM_FUNC_MAP[key] if kind == 'param' else self.PROCESSOR_MAP[key]
        if func is None:
            print(f"{key} 未定义，无法验证")
            return
        results = []
        for i, case in enumerate(self.test_cases.get(key, [{}])):
            try:
                if isinstance(case, dict):
                    result = func(**case)
                elif isinstance(case, (list, tuple)):
                    result = func(*case)
                else:
                    result = func(case)
                results.append({'case': case, 'result': result, 'success': True})
            except Exception as e:
                results.append({'case': case, 'error': str(e), 'success': False})
        self.execution_results[key] = results

    def process_plan(self, plan_json):

        param_keys = set()
        processor_keys = set()
        for task in plan_json.get('tasks', []):
            for param in task.get('param_funcs', []):
                param_keys.add(param)
            processor = task.get('task_result_processor')
            if processor:
                processor_keys.add(processor)
        # 生成并注入代码
        for key in param_keys:
            self.generate_and_inject_code(key, kind='param')
        for key in processor_keys:
            self.generate_and_inject_code(key, kind='processor')
        # 验证
        for key in param_keys:
            self.verify_and_run(key, kind='param')
        for key in processor_keys:
            self.verify_and_run(key, kind='processor')
        # 返回所有
        return {
            'generated_code': self.generated_code,
            'execution_results': self.execution_results
        }

    def auto_debug_and_execute(self, plan_json, debug_agent, max_rounds=5):
        for _ in range(max_rounds):
            result = self.process_plan(plan_json)
            has_error = False
            # 检查所有执行结果
            for key, cases in result['execution_results'].items():
                for case_result in cases:
                    if not case_result.get('success', True):
                        has_error = True
                        # debug
                        code = result['generated_code'][key]
                        error = case_result.get('error')
                        context = case_result.get('case')
                        print(f"\n检测到 {key} 报错：{error}\n正在请求DebugAgent修复……")
                        fixed_code = debug_agent.debug_code(key, code, error, context)
                        # 重新注入修复后的代码
                        self.generated_code[key] = fixed_code
                        local_env = {}
                        try:
                            exec(fixed_code, {}, local_env)
                            func = local_env.get(key)
                            if func:
                                if key in self.PARAM_FUNC_MAP:
                                    self.PARAM_FUNC_MAP[key] = func
                                elif key in self.PROCESSOR_MAP:
                                    self.PROCESSOR_MAP[key] = func
                        except Exception as e:
                            print(f"修复后代码依然有问题: {key}\n{fixed_code}\n{e}")
            if not has_error:
                print("所有代码已通过验证，无报错！")
                return result
        print(f"达到最大debug轮数{max_rounds}，仍有报错。")
        return result
def simple_llm_caller(prompt):
    # 实际上应该调用你的 LLM API
    # 这里只做简单模拟
    if 'star_count' in prompt:
        return "def star_count(repo):\n    return 1234"
    if 'fork_count' in prompt:
        return "def fork_count(repo):\n    return 567"
    if 'parse_github_stats' in prompt:
        return "def parse_github_stats(search_results):\n    # 假设search_results为字符串\n    return {'star_count': 1234, 'fork_count': 567}"
    if 'save_python_file' in prompt:
        return "def save_python_file(script_content):\n    with open('test.py', 'w') as f:\n        f.write(script_content)\n    return 'test.py'"
    # 默认返回一个空函数
    return f"def {prompt.split('函数名为 ')[1].split('，')[0]}(*args, **kwargs):\n    return None"

# 假设plan_json如下（可替换为实际plan结果）
plan_json = {
    'tasks': [
        {'param_funcs': ['star_count', 'fork_count']},
        {'task_result_processor': 'parse_github_stats'},
        {'task_result_processor': 'save_python_file'}
    ]
}

if __name__ == "__main__":
    agent = ExecutionAgent(
        param_func_map={},
        processor_map={},
        llm_caller=simple_llm_caller,
        test_cases={
            'star_count': [{'repo': 'camel-ai/camel'}],
            'fork_count': [{'repo': 'camel-ai/camel'}],
            'parse_github_stats': [{'search_results': 'xxx'}],
            'save_python_file': [{'script_content': 'print(1)'}]
        }
    )
    result = agent.process_plan(plan_json)
    print("所有LLM生成代码：")
    for k, v in result['generated_code'].items():
        print(f"--- {k} ---\n{v}\n")
    print("所有执行结果：")
    for k, v in result['execution_results'].items():
        print(f"--- {k} ---\n{v}\n")