import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import subprocess
import os
import sys
import threading
import time
from datetime import datetime
from PIL import Image, ImageTk

def resource_path(relative_path):
    """ 获取资源绝对路径，用于 PyInstaller 打包 """
    try:
        # PyInstaller 创建的临时目录
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class CloudflareGenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cloudflare AI 图片生成器")
        self.root.geometry("900x700")
        
        # 配置样式
        style = ttk.Style()
        style.theme_use('clam')
        
        # 主容器
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 输出目录
        self.output_dir = os.path.join(os.getcwd(), "output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 路径选择区域
        path_frame = ttk.Frame(main_frame)
        path_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(path_frame, text="保存路径:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value=self.output_dir)
        self.path_entry = ttk.Entry(path_frame, textvariable=self.path_var, state='readonly')
        self.path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.browse_btn = ttk.Button(path_frame, text="浏览...", command=self.browse_output_dir)
        self.browse_btn.pack(side=tk.LEFT)
        
        # 选项卡
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 单图生成页面
        self.single_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.single_frame, text=' 单图生成 ')
        self.setup_single_tab()
        
        # 批量生成页面
        self.batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_frame, text=' 批量生成 ')
        self.setup_batch_tab()
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def browse_output_dir(self):
        selected_dir = filedialog.askdirectory()
        if selected_dir:
            self.output_dir = selected_dir
            self.path_var.set(self.output_dir)

    def setup_single_tab(self):
        frame = ttk.Frame(self.single_frame, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 输入区
        input_frame = ttk.LabelFrame(frame, text="提示词", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.single_prompt = tk.Text(input_frame, height=3, width=50)
        self.single_prompt.pack(fill=tk.X, padx=5, pady=5)
        self.single_prompt.insert("1.0", "cyberpunk city, neon lights, rain, high detail")
        
        # 按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.generate_btn = ttk.Button(btn_frame, text="生成图片", command=self.start_single_generation)
        self.generate_btn.pack(side=tk.LEFT)
        
        # 图片预览区
        preview_frame = ttk.LabelFrame(frame, text="预览", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(preview_frame, text="暂无图片")
        self.image_label.pack(expand=True)

    def setup_batch_tab(self):
        frame = ttk.Frame(self.batch_frame, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 说明
        ttk.Label(frame, text="每行输入一个提示词:").pack(anchor=tk.W)
        
        # 输入区
        self.batch_prompts = scrolledtext.ScrolledText(frame, height=10)
        self.batch_prompts.pack(fill=tk.BOTH, expand=True, pady=5)
        self.batch_prompts.insert("1.0", "cat in space suit\ndog playing poker\nflying car in future city")
        
        # 按钮和进度
        ctrl_frame = ttk.Frame(frame)
        ctrl_frame.pack(fill=tk.X, pady=10)
        
        self.batch_btn = ttk.Button(ctrl_frame, text="开始批量生成", command=self.start_batch_generation)
        self.batch_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.progress = ttk.Progressbar(ctrl_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 日志区
        log_frame = ttk.LabelFrame(frame, text="生成日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.status_var.set(message)

    def run_node_generator(self, prompt, output_path):
        try:
            # 获取 generator.js 的路径（兼容打包后的环境）
            script_path = resource_path("generator.js")
            
            # 调用 node 脚本
            cmd = ["node", script_path, "--prompt", prompt, "--output", output_path]
            # 创建无窗口的启动信息（仅 Windows）
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', startupinfo=startupinfo)
            
            if result.returncode != 0:
                return False, result.stderr
            return True, output_path
        except Exception as e:
            return False, str(e)

    def start_single_generation(self):
        prompt = self.single_prompt.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("提示", "请输入提示词")
            return
            
        self.generate_btn.config(state='disabled')
        self.status_var.set("正在生成...")
        
        thread = threading.Thread(target=self._single_gen_thread, args=(prompt,))
        thread.start()

    def get_filename_from_prompt(self, prompt):
        # 取第一个逗号前的部分作为主要描述
        main_part = prompt.split(',')[0].strip()
        # 替换非法字符和空格
        safe_name = "".join([c if c.isalnum() or c in (' ', '-', '_') else '' for c in main_part])
        safe_name = safe_name.replace(' ', '_')
        # 限制长度
        if len(safe_name) > 30:
            safe_name = safe_name[:30]
        
        # 如果为空（全是特殊字符），使用默认名
        if not safe_name:
            safe_name = "image"
            
        return safe_name

    def _single_gen_thread(self, prompt):
        name_base = self.get_filename_from_prompt(prompt)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name_base}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        success, msg = self.run_node_generator(prompt, filepath)
        
        self.root.after(0, self._single_gen_complete, success, msg, filepath)

    def _single_gen_complete(self, success, msg, filepath):
        self.generate_btn.config(state='normal')
        if success:
            self.status_var.set("生成成功")
            self.show_image(filepath)
        else:
            self.status_var.set("生成失败")
            messagebox.showerror("错误", f"生成失败: {msg}")

    def show_image(self, path):
        try:
            img = Image.open(path)
            # 调整图片大小以适应窗口，保持比例
            display_size = (500, 500)
            img.thumbnail(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo # 保持引用
        except Exception as e:
            self.image_label.config(text=f"无法加载图片: {e}")

    def start_batch_generation(self):
        prompts = self.batch_prompts.get("1.0", tk.END).strip().split('\n')
        prompts = [p.strip() for p in prompts if p.strip()]
        
        if not prompts:
            messagebox.showwarning("提示", "请输入至少一个提示词")
            return
            
        self.batch_btn.config(state='disabled')
        self.progress['maximum'] = len(prompts)
        self.progress['value'] = 0
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        
        thread = threading.Thread(target=self._batch_gen_thread, args=(prompts,))
        thread.start()

    def _batch_gen_thread(self, prompts):
        for i, prompt in enumerate(prompts):
            self.root.after(0, self.log, f"正在生成 ({i+1}/{len(prompts)}): {prompt}")
            
            name_base = self.get_filename_from_prompt(prompt)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name_base}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            success, msg = self.run_node_generator(prompt, filepath)
            
            if success:
                self.root.after(0, self.log, f"成功: {filename}")
            else:
                self.root.after(0, self.log, f"失败: {msg}")
            
            self.root.after(0, self._update_progress, i + 1)
            
        self.root.after(0, self._batch_gen_complete)

    def _update_progress(self, val):
        self.progress['value'] = val

    def _batch_gen_complete(self):
        self.batch_btn.config(state='normal')
        self.status_var.set("批量任务完成")
        messagebox.showinfo("完成", "批量生成任务已完成")

if __name__ == "__main__":
    root = tk.Tk()
    app = CloudflareGenApp(root)
    root.mainloop()
