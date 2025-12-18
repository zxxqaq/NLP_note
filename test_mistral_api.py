#!/usr/bin/env python3
"""
测试 Mistral API 引擎
"""

import os
import sys

# 添加 DEXTER 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DEXTER-macos'))

from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator

def test_mistral_api():
    """测试 Mistral API 连接和调用"""
    
    # 检查 API Key
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("✗ 错误: MISTRAL_API_KEY 环境变量未设置")
        print("请运行: export MISTRAL_API_KEY='your_api_key'")
        return False
    
    print("=" * 60)
    print("测试 Mistral API 引擎")
    print("=" * 60)
    print(f"API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
    print(f"RPM 限制: {os.environ.get('MISTRAL_RPM_LIMIT', '60')} 请求/分钟")
    print()
    
    try:
        # 创建引擎实例
        print("[1/3] 创建 Mistral 引擎实例...")
        config_instance = LLMEngineOrchestrator()
        llm_engine = config_instance.get_llm_engine(
            data="",
            llm_class="mistral",
            model_name="mistralai/Mistral-7B-Instruct-v0.1"  # 会自动转换为 API 格式
        )
        print("✓ 引擎创建成功")
        print()
        
        # 测试 API 调用
        print("[2/3] 测试 API 调用...")
        system_prompt = "You are a helpful assistant. Answer the question concisely."
        user_prompt = "What is the capital of France? Answer in one word."
        
        print(f"System Prompt: {system_prompt}")
        print(f"User Prompt: {user_prompt}")
        print("正在调用 API...")
        
        response = llm_engine.get_mistral_completion(system_prompt, user_prompt)
        
        print("✓ API 调用成功")
        print(f"响应: {response}")
        print()
        
        # 测试 RPM 限制
        print("[3/3] 测试 RPM 限制（连续调用2次）...")
        import time
        start_time = time.time()
        
        response1 = llm_engine.get_mistral_completion(
            "You are a helpful assistant.",
            "Say 'Hello' in one word."
        )
        time1 = time.time() - start_time
        
        start_time2 = time.time()
        response2 = llm_engine.get_mistral_completion(
            "You are a helpful assistant.",
            "Say 'World' in one word."
        )
        time2 = time.time() - start_time2
        
        print(f"✓ 第一次调用耗时: {time1:.2f} 秒")
        print(f"✓ 第二次调用耗时: {time2:.2f} 秒")
        print(f"✓ 响应1: {response1}")
        print(f"✓ 响应2: {response2}")
        
        if time2 >= llm_engine.min_interval:
            print(f"✓ RPM 限制正常工作（间隔 >= {llm_engine.min_interval:.2f} 秒）")
        else:
            print(f"⚠ RPM 限制可能未生效（间隔 < {llm_engine.min_interval:.2f} 秒）")
        
        print()
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mistral_api()
    sys.exit(0 if success else 1)




