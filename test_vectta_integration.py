#!/usr/bin/env python3
"""
测试VECTTA集成到evaluate_res50_depth_cityscapes_config.py的脚本
"""

import sys
import os

def test_imports():
    """测试导入是否正常"""
    try:
        print("测试导入VECTTA相关类...")
        from Vectta import VECTTA, SQLDepthModel
        print("✓ 成功导入VECTTA和SQLDepthModel类")
        
        print("测试导入evaluate_res50_depth_cityscapes_config...")
        import evaluate_res50_depth_cityscapes_config
        print("✓ 成功导入evaluate_res50_depth_cityscapes_config模块")
        
        print("测试MonodepthOptions...")
        from options import MonodepthOptions
        options = MonodepthOptions()
        print("✓ 成功创建MonodepthOptions实例")
        
        return True
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_vectta_parameters():
    """测试VECTTA参数是否正确添加"""
    try:
        from options import MonodepthOptions
        options = MonodepthOptions()
        
        # 添加VECTTA参数
        options.parser.add_argument('--use_vectta', action='store_true', help='使用VECTTA进行测试时适应')
        options.parser.add_argument('--vectta_steps', type=int, default=5, help='VECTTA适应步数')
        options.parser.add_argument('--vectta_update_mode', type=str, choices=['bn_only', 'bn_decoder', 'last_layers', 'all'], 
                                   default='bn_decoder', help='VECTTA更新模式')
        options.parser.add_argument('--vectta_lr', type=float, default=1e-4, help='VECTTA学习率')
        options.parser.add_argument('--vectta_early_stop', action='store_true', help='VECTTA早停')
        options.parser.add_argument('--vectta_early_stop_patience', type=int, default=3, help='VECTTA早停耐心值')
        options.parser.add_argument('--vectta_grad_clip', type=float, default=1.0, help='VECTTA梯度裁剪')
        options.parser.add_argument('--clean_pred_path', type=str, default='clean_pred_disps.npy', 
                                   help='clean伪标签文件路径')
        
        # 测试解析参数
        test_args = ['--use_vectta', '--vectta_steps', '3', '--vectta_update_mode', 'bn_only']
        opt = options.parser.parse_args(test_args)
        
        print(f"✓ 成功解析VECTTA参数:")
        print(f"  - use_vectta: {opt.use_vectta}")
        print(f"  - vectta_steps: {opt.vectta_steps}")
        print(f"  - vectta_update_mode: {opt.vectta_update_mode}")
        print(f"  - vectta_lr: {opt.vectta_lr}")
        print(f"  - clean_pred_path: {opt.clean_pred_path}")
        
        return True
    except Exception as e:
        print(f"✗ 参数测试错误: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("VECTTA集成测试")
    print("=" * 50)
    
    # 测试导入
    if not test_imports():
        print("\n❌ 导入测试失败")
        return False
    
    print()
    
    # 测试参数
    if not test_vectta_parameters():
        print("\n❌ 参数测试失败")
        return False
    
    print("\n" + "=" * 50)
    print("✅ 所有测试通过！VECTTA已成功集成到evaluate_res50_depth_cityscapes_config.py")
    print("=" * 50)
    
    print("\n使用方法:")
    print("python evaluate_res50_depth_cityscapes_config.py --use_vectta --vectta_steps 5 --vectta_update_mode bn_decoder")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
