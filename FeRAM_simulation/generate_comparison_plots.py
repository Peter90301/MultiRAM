#!/usr/bin/env python3
"""
FeRAM vs GPU 性能与能效可视化对比
生成对比图表
"""
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据
hardware = ['FeRAM\n(SK hynix)', 'FeRAM\n(Kookmin)', 'NVIDIA\nA100', 'NVIDIA\nH100']
energy_efficiency = [224.4, 405, 1.56, 5.65]  # TOPS/W (Kookmin取中值)
power_consumption = [1.2, 0.8, 400, 700]  # Watts
compute_power = [0.26, 0.32, 624, 3958]  # TOPS
colors = ['#2E7D32', '#43A047', '#1565C0', '#0D47A1']

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('FeRAM vs NVIDIA GPU Performance Comparison', fontsize=16, fontweight='bold')

# 1. 能效对比（TOPS/W）- 对数尺度
ax1 = axes[0, 0]
bars1 = ax1.bar(hardware, energy_efficiency, color=colors, alpha=0.8, edgecolor='black')
ax1.set_yscale('log')
ax1.set_ylabel('Energy Efficiency (TOPS/W)', fontsize=12, fontweight='bold')
ax1.set_title('Energy Efficiency (Higher is Better)', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars1, energy_efficiency)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height * 1.2,
             f'{val:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# 添加能效比标注
ax1.text(0.5, 200, '144x', ha='center', fontsize=9, color='red', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax1.text(2.5, 20, '40x', ha='center', fontsize=9, color='red', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# 2. 功耗对比（W）- 对数尺度
ax2 = axes[0, 1]
bars2 = ax2.bar(hardware, power_consumption, color=colors, alpha=0.8, edgecolor='black')
ax2.set_yscale('log')
ax2.set_ylabel('Power Consumption (W)', fontsize=12, fontweight='bold')
ax2.set_title('Power Consumption (Lower is Better)', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars2, power_consumption)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height * 1.2,
             f'{val:.1f}W',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# 3. 算力对比（TOPS）- 对数尺度
ax3 = axes[1, 0]
bars3 = ax3.bar(hardware, compute_power, color=colors, alpha=0.8, edgecolor='black')
ax3.set_yscale('log')
ax3.set_ylabel('Computing Power (TOPS)', fontsize=12, fontweight='bold')
ax3.set_title('Raw Computing Power', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars3, compute_power)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height * 1.2,
             f'{val:.2f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# 4. 综合对比雷达图
ax4 = axes[1, 1]
ax4.remove()
ax4 = fig.add_subplot(224, projection='polar')

# 雷达图数据（归一化到0-1）
categories = ['Energy\nEfficiency', 'Low\nPower', 'Computing\nPower', 'Cost\nEfficiency']
N = len(categories)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# FeRAM (归一化: 能效高=1, 功耗低=1, 算力=相对值, 成本效益高=1)
feram_values = [1.0, 1.0, 0.001, 0.9]  # FeRAM优势
gpu_h100_values = [0.025, 0.002, 1.0, 0.3]  # H100优势
feram_values += feram_values[:1]
gpu_h100_values += gpu_h100_values[:1]

ax4.plot(angles, feram_values, 'o-', linewidth=2, label='FeRAM', color='#2E7D32')
ax4.fill(angles, feram_values, alpha=0.25, color='#2E7D32')

ax4.plot(angles, gpu_h100_values, 'o-', linewidth=2, label='H100', color='#0D47A1')
ax4.fill(angles, gpu_h100_values, alpha=0.25, color='#0D47A1')

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=10)
ax4.set_ylim(0, 1)
ax4.set_yticks([0.25, 0.5, 0.75, 1.0])
ax4.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
ax4.set_title('Comprehensive Comparison\n(Normalized)', fontsize=13, fontweight='bold', pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax4.grid(True)

plt.tight_layout()
plt.savefig('feram_vs_gpu_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存: feram_vs_gpu_comparison.png")

# 创建第二张图：运行时间和能量消耗对比
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Performance Metrics for Clustering Task (786,432 Spectra)', 
              fontsize=15, fontweight='bold')

# 运行时间对比
ax_time = axes2[0]
platforms = ['CPU\nSimulation', 'FeRAM\nHardware', 'NVIDIA\nA100', 'NVIDIA\nH100']
runtime = [648, 51, 27, 4.2]  # seconds
speedup = [1, 12.7, 24, 154]
colors2 = ['#757575', '#2E7D32', '#1565C0', '#0D47A1']

bars_time = ax_time.bar(platforms, runtime, color=colors2, alpha=0.8, edgecolor='black')
ax_time.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
ax_time.set_title('Execution Time (Lower is Better)', fontsize=13, fontweight='bold')
ax_time.set_yscale('log')
ax_time.grid(axis='y', alpha=0.3, linestyle='--')

for i, (bar, time, speed) in enumerate(zip(bars_time, runtime, speedup)):
    height = bar.get_height()
    ax_time.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                f'{time:.1f}s\n({speed:.1f}x)',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

# 能量消耗对比
ax_energy = axes2[1]
energy = [25920, 61, 10800, 2940]  # Joules (CPU假设40W功耗)
bars_energy = ax_energy.bar(platforms, energy, color=colors2, alpha=0.8, edgecolor='black')
ax_energy.set_ylabel('Energy Consumption (Joules)', fontsize=12, fontweight='bold')
ax_energy.set_title('Total Energy (Lower is Better)', fontsize=13, fontweight='bold')
ax_energy.set_yscale('log')
ax_energy.grid(axis='y', alpha=0.3, linestyle='--')

for i, (bar, eng) in enumerate(zip(bars_energy, energy)):
    height = bar.get_height()
    if eng >= 1000:
        label = f'{eng/1000:.1f}kJ'
    else:
        label = f'{eng:.0f}J'
    ax_energy.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                  label,
                  ha='center', va='bottom', fontweight='bold', fontsize=9)

# 添加能量节省标注
ax_energy.text(0.5, 15000, '425x', ha='center', fontsize=9, color='red', fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('feram_vs_gpu_runtime_energy.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存: feram_vs_gpu_runtime_energy.png")

# 打印数值摘要
print("\n" + "="*80)
print("关键性能指标摘要")
print("="*80)
print(f"\n能效对比:")
print(f"  FeRAM (SK hynix):     {energy_efficiency[0]:.1f} TOPS/W")
print(f"  FeRAM (Kookmin):      {energy_efficiency[1]:.0f} TOPS/W (avg)")
print(f"  NVIDIA A100:          {energy_efficiency[2]:.2f} TOPS/W")
print(f"  NVIDIA H100:          {energy_efficiency[3]:.2f} TOPS/W")
print(f"\n能效比:")
print(f"  FeRAM vs A100:        {energy_efficiency[0]/energy_efficiency[2]:.0f}x 更节能")
print(f"  FeRAM vs H100:        {energy_efficiency[0]/energy_efficiency[3]:.0f}x 更节能")

print(f"\n运行时间对比（786,432光谱）:")
print(f"  CPU Simulation:       {runtime[0]:.0f} 秒")
print(f"  FeRAM Hardware:       {runtime[1]:.0f} 秒 (推测)")
print(f"  NVIDIA A100:          {runtime[2]:.0f} 秒 (推测)")
print(f"  NVIDIA H100:          {runtime[3]:.1f} 秒 (推测)")

print(f"\n加速比 (vs CPU):")
print(f"  FeRAM:                {speedup[1]:.1f}x")
print(f"  A100:                 {speedup[2]:.0f}x")
print(f"  H100:                 {speedup[3]:.0f}x")

print(f"\n能量消耗对比:")
print(f"  CPU:                  {energy[0]:,.0f} J")
print(f"  FeRAM:                {energy[1]:,.0f} J")
print(f"  A100:                 {energy[2]:,.0f} J")
print(f"  H100:                 {energy[3]:,.0f} J")

print(f"\n能量节省 (vs GPU):")
print(f"  FeRAM vs A100:        {energy[2]/energy[1]:.0f}x")
print(f"  FeRAM vs H100:        {energy[3]/energy[1]:.0f}x")

plt.show()
