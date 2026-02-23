% 加载 .mat 文件中的数据
data = load('your_name.mat');
boundary_3d = int8(data.boundary_3d);

% 获取数据的大小
[inline_size, crossline_size, time_size] = size(boundary_3d);

% 获取变化位置的索引
[inline_idx, crossline_idx, time_idx] = ind2sub(size(boundary_3d), find(boundary_3d == 1));

% 创建 3D 散点图
figure;
scatter3(inline_idx, crossline_idx, time_idx, 5, time_idx, 'filled');
z_min = 20
z_max = 250

set(gca, 'ZDir', 'reverse'); % 使时间从上到下
zlim([z_min, z_max]); % 设定 z 轴范围
colormap(parula); % 选择更明显的 colormap，比如 jet 或 parula
caxis([z_min, z_max]); % 让颜色范围与 Z 轴范围匹配
colorbar; % 显示颜色条，方便查看颜色对应的值
view(135, 45); % 调整到对向视角