import gradio as gr
import pandas as pd
import torch
import os
from PIL import Image
import warnings
import yfinance as yf
from stock_prediction_lstm import predict, format_feature
from RLagent import process_stock
from datetime import datetime
from process_stock_data import get_stock_data, clean_csv_files

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'tmp/gradio'

# 创建所有必要的目录
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('tmp/gradio/pic', exist_ok=True)
os.makedirs('tmp/gradio/pic/predictions', exist_ok=True)
os.makedirs('tmp/gradio/pic/loss', exist_ok=True)
os.makedirs('tmp/gradio/pic/earnings', exist_ok=True)
os.makedirs('tmp/gradio/pic/trades', exist_ok=True)
os.makedirs('tmp/gradio/models', exist_ok=True)
os.makedirs('tmp/gradio/transactions', exist_ok=True)
os.makedirs('tmp/gradio/ticker', exist_ok=True)

def get_data(ticker, start_date, end_date, progress=gr.Progress()):
    data_folder = 'tmp/gradio/ticker'
    temp_path = f'{data_folder}/{ticker}.csv'
    try:        
        # 获取并保存所有股票数据
        progress(0, desc="开始获取股票数据...")
        stock_data = get_stock_data(ticker, start_date, end_date)
        progress(0.4, desc="计算技术指标...")
        stock_data.to_csv(temp_path)
        progress(0.7, desc="处理数据格式...")
        clean_csv_files(temp_path)
        progress(1.0, desc="数据获取完成")
        return temp_path, "数据获取成功"
    except Exception as e:
        return None, f"获取数据出错: {str(e)}"

def validate_and_fix_data(file_path):
    """验证并修复股票数据文件，确保其符合预期格式"""
    try:
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 检查必要的列是否存在
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"警告: 文件 {file_path} 缺少必要的列: {', '.join(missing_cols)}")
            # 如果缺少必要列，创建默认数据
            dates = pd.date_range(start='2020-01-01', periods=len(df) if not df.empty else 100)
            default_df = pd.DataFrame({
                'Date': dates,
                'Open': [100] * len(dates),
                'High': [110] * len(dates),
                'Low': [90] * len(dates),
                'Close': [105] * len(dates),
                'Volume': [1000000] * len(dates)
            })
            default_df.to_csv(file_path, index=False)
            print(f"已为 {file_path} 创建默认数据")
            return False
        
        # 检查是否有NaN值
        if df[required_cols].isna().any().any():
            print(f"警告: 文件 {file_path} 包含NaN值，将进行填充")
            # 填充NaN值
            for col in required_cols:
                if df[col].isna().any():
                    # 使用前向填充
                    df[col] = df[col].fillna(method='ffill')
                    # 然后使用后向填充（处理开头的NaN）
                    df[col] = df[col].fillna(method='bfill')
                    # 最后使用列平均值填充任何剩余的NaN
                    df[col] = df[col].fillna(df[col].mean() if not df[col].empty else 0)
            
            # 保存修复后的数据
            df.to_csv(file_path, index=False)
            print(f"已修复 {file_path} 中的NaN值")
        
        # 检查数据长度是否足够
        if len(df) < 100:  # 假设至少需要100行数据
            print(f"警告: 文件 {file_path} 的数据量不足 ({len(df)} 行)")
            # 可以选择通过复制已有数据来扩充数据集
            
        return True
    
    except Exception as e:
        print(f"验证和修复文件 {file_path} 时出错: {str(e)}")
        # 创建默认数据
        dates = pd.date_range(start='2020-01-01', periods=100)
        default_df = pd.DataFrame({
            'Date': dates,
            'Open': [100] * 100,
            'High': [110] * 100,
            'Low': [90] * 100,
            'Close': [105] * 100,
            'Volume': [1000000] * 100
        })
        default_df.to_csv(file_path, index=False)
        print(f"已为 {file_path} 创建默认数据")
        return False

def process_and_predict(temp_csv_path, epochs, batch_size, learning_rate, 
                       window_size, initial_money, agent_iterations, save_dir, progress=gr.Progress()):
    if not temp_csv_path:
        return [None] * 9  # 返回空结果
        
    try:
        # 从文件路径中提取股票代码（去掉.csv后缀）
        ticker = os.path.basename(temp_csv_path).split('.')[0]
        
        progress(0.05, desc="正在加载并验证股票数据...")
        # 验证并修复数据
        data_valid = validate_and_fix_data(temp_csv_path)
        if not data_valid:
            progress(0.1, desc="使用替代数据...")
        
        # 读取数据
        stock_data = pd.read_csv(temp_csv_path)
        
        # 确保处理特征前检查数据结构
        try:
            stock_features = format_feature(stock_data)
        except Exception as e:
            print(f"格式化特征时出错: {str(e)}")
            # 创建一个具有默认格式的提示
            return [None] * 9
        
        progress(0.1, desc="开始LSTM预测训练...")
        # 使用纯股票代码而非文件名
        try:
            metrics = predict(
                save_dir=save_dir,
                ticker_name=ticker,
                stock_data=stock_data,
                stock_features=stock_features,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
        except Exception as e:
            print(f"LSTM预测训练出错: {str(e)}")
            metrics = {'accuracy': 0, 'rmse': 0, 'mae': 0}
        
        progress(0.5, desc="开始交易代理训练...")
        # 使用纯股票代码而非文件名
        try:
            trading_results = process_stock(
                ticker,
                save_dir,
                window_size=window_size,
                initial_money=initial_money,
                iterations=agent_iterations
            )
        except Exception as e:
            print(f"交易代理训练出错: {str(e)}")
            trading_results = {'total_gains': 0, 'investment_return': 0, 'trades_buy': 0, 'trades_sell': 0}
        
        progress(0.9, desc="生成结果可视化...")
        # 使用安全的图像加载方式
        images = []
        try:
            prediction_path = f"{save_dir}/pic/predictions/{ticker}_prediction.png"
            if os.path.exists(prediction_path):
                prediction_plot = Image.open(prediction_path)
                images.append(prediction_plot)
            else:
                print(f"无法找到预测图片: {prediction_path}")
                # 创建一个空白图像作为替代
                blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
                images.append(blank_img)
        except Exception as e:
            print(f"加载预测图片出错: {str(e)}")
            blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
            images.append(blank_img)
        
        try:
            loss_path = f"{save_dir}/pic/loss/{ticker}_loss.png"
            if os.path.exists(loss_path):
                loss_plot = Image.open(loss_path)
                images.append(loss_plot)
            else:
                print(f"无法找到损失图片: {loss_path}")
                blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
                images.append(blank_img)
        except Exception as e:
            print(f"加载损失图片出错: {str(e)}")
            blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
            images.append(blank_img)
            
        try:
            earnings_path = f"{save_dir}/pic/earnings/{ticker}_cumulative.png"
            if os.path.exists(earnings_path):
                earnings_plot = Image.open(earnings_path)
                images.append(earnings_plot)
            else:
                print(f"无法找到收益图片: {earnings_path}")
                blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
                images.append(blank_img)
        except Exception as e:
            print(f"加载收益图片出错: {str(e)}")
            blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
            images.append(blank_img)
            
        try:
            trades_path = f"{save_dir}/pic/trades/{ticker}_trades.png"
            if os.path.exists(trades_path):
                trades_plot = Image.open(trades_path)
                images.append(trades_plot)
            else:
                print(f"无法找到交易图片: {trades_path}")
                blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
                images.append(blank_img)
        except Exception as e:
            print(f"加载交易图片出错: {str(e)}")
            blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
            images.append(blank_img)
        
        try:
            transactions_path = f"{save_dir}/transactions/{ticker}_transactions.csv"
            if os.path.exists(transactions_path):
                transactions_df = pd.read_csv(transactions_path)
            else:
                print(f"无法找到交易记录: {transactions_path}")
                # 创建空的交易记录
                transactions_df = pd.DataFrame(columns=['day', 'operate', 'price', 'investment', 'total_balance'])
        except Exception as e:
            print(f"加载交易记录出错: {str(e)}")
            transactions_df = pd.DataFrame(columns=['day', 'operate', 'price', 'investment', 'total_balance'])
        
        progress(1.0, desc="完成!")
        return [
            images,
            metrics['accuracy'] * 100 if 'accuracy' in metrics else 0,
            metrics['rmse'] if 'rmse' in metrics else 0,
            metrics['mae'] if 'mae' in metrics else 0,
            trading_results['total_gains'] if 'total_gains' in trading_results else 0,
            trading_results['investment_return'] if 'investment_return' in trading_results else 0,
            trading_results['trades_buy'] if 'trades_buy' in trading_results else 0,
            trading_results['trades_sell'] if 'trades_sell' in trading_results else 0,
            transactions_df
        ]
    except Exception as e:
        print(f"处理错误: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细错误堆栈
        
        # 返回空结果但带有有意义的错误信息
        blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
        # 在空白图像上添加错误信息
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(blank_img)
        draw.text((50, 240), f"处理出错: {str(e)}", fill=(0, 0, 0))
        
        return [
            [blank_img, blank_img, blank_img, blank_img],
            0, 0, 0, 0, 0, 0, 0,
            pd.DataFrame(columns=['day', 'operate', 'price', 'investment', 'total_balance'])
        ]

with gr.Blocks() as demo:
    gr.Markdown("# 智能股票预测与交易Agent")
    
    save_dir_state = gr.State(value='tmp/gradio')
    temp_csv_state = gr.State(value=None)
    
    with gr.Row():
        with gr.Column(scale=2):
            ticker_input = gr.Textbox(label="股票代码 (例如: AAPL)")
        with gr.Column(scale=2):
            start_date = gr.Textbox(
                label="开始日期 (YYYY-MM-DD)", 
                value=(datetime.now().replace(year=datetime.now().year-4).strftime('%Y-%m-%d'))
            )
        with gr.Column(scale=2):
            end_date = gr.Textbox(
                label="结束日期 (YYYY-MM-DD)", 
                value=datetime.now().strftime('%Y-%m-%d')
            )
        with gr.Column(scale=1):
            fetch_button = gr.Button("获取数据")
    
    with gr.Row():
        status_output = gr.Textbox(label="状态信息", interactive=False)
    
    with gr.Row():
        data_file = gr.File(label="下载股票数据", visible=True, interactive=False)
    
    with gr.Tabs():
        with gr.TabItem("LSTM预测参数"):
            with gr.Column():
                lstm_epochs = gr.Slider(minimum=100, maximum=1000, value=500, step=10, 
                                      label="LSTM训练轮数")
                lstm_batch = gr.Slider(minimum=16, maximum=128, value=32, step=16, 
                                     label="LSTM批次大小")
                learning_rate = gr.Slider(minimum=0.0001, maximum=0.01, value=0.001, 
                                        step=0.0001, label="LSTM训练学习率")
        
        with gr.TabItem("交易代理参数"):
            with gr.Column():
                window_size = gr.Slider(minimum=10, maximum=100, value=30, step=5,
                                      label="时间窗口大小")
                initial_money = gr.Number(value=10000, label="初始投资金额 ($)")
                agent_iterations = gr.Slider(minimum=100, maximum=1000, value=500, 
                                          step=50, label="代理训练迭代次数")
    
    with gr.Row():
        train_button = gr.Button("开始训练", interactive=False)
    
    with gr.Row():
        output_gallery = gr.Gallery(label="分析结果可视化", show_label=True,
                                  elem_id="gallery", columns=4, rows=1,
                                  height="auto", object_fit="contain")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 预测指标")
            accuracy_output = gr.Number(label="预测准确率 (%)")
            rmse_output = gr.Number(label="RMSE (均方根误差)")
            mae_output = gr.Number(label="MAE (平均绝对误差)")
        
        with gr.Column(scale=1):
            gr.Markdown("### 交易指标")
            gains_output = gr.Number(label="总收益 ($)")
            return_output = gr.Number(label="投资回报率 (%)")
            trades_buy_output = gr.Number(label="买入次数")
            trades_sell_output = gr.Number(label="卖出次数")
    
    with gr.Row():
        gr.Markdown("### 交易记录")
        transactions_df = gr.DataFrame(
            headers=["day", "operate", "price", "investment", "total_balance"],
            label="交易详细记录"
        )
    
    def update_interface(csv_path):
        return (
            csv_path if csv_path else None,  # 更新文件下载
            gr.update(interactive=bool(csv_path))  # 更新训练按钮
        )
    
    # 获取数据按钮事件
    fetch_result = fetch_button.click(
        fn=get_data,
        inputs=[ticker_input, start_date, end_date],
        outputs=[temp_csv_state, status_output]
    )
    
    # 更新界面状态
    fetch_result.then(
        update_interface,
        inputs=[temp_csv_state],
        outputs=[data_file, train_button]
    )
    
    # 训练按钮事件
    train_button.click(
        fn=process_and_predict,
        inputs=[
            temp_csv_state,
            lstm_epochs,
            lstm_batch,
            learning_rate,
            window_size,
            initial_money,
            agent_iterations,
            save_dir_state
        ],
        outputs=[
            output_gallery,
            accuracy_output,
            rmse_output,
            mae_output,
            gains_output,
            return_output,
            trades_buy_output,
            trades_sell_output,
            transactions_df
        ]
    )

demo.launch(server_port=7860, share=True)