#!/bin/bash

# 设置工作目录
CODE_ROOT="code_root"
cd "$CODE_ROOT" || exit 1

# 重命名文件夹
rename_directories() {
    # 使用find命令查找所有目录
    find . -type d | while read -r dir; do
        new_dir="$dir"
        # 替换文件夹名称
        new_dir=$(echo "$new_dir" | sed 's/admin/admin/g')
        new_dir=$(echo "$new_dir" | sed 's/user/user/g')
        new_dir=$(echo "$new_dir" | sed 's/user/user/g')
        
        # 如果文件夹名称有变化，则重命名
        if [ "$dir" != "$new_dir" ]; then
            mv "$dir" "$new_dir"
            echo "Renamed directory: $dir -> $new_dir"
        fi
    done
}

# 修改文件内容
modify_file_contents() {
    # 使用find命令查找所有文件
    find . -type f | while read -r file; do
        # 检查文件是否是文本文件
        if file "$file" | grep -q "text"; then
            # 创建临时文件
            temp_file=$(mktemp)
            
            # 替换文件内容
            sed 's/admin/admin/g' "$file" | \
            sed 's/user/user/g' | \
            sed 's/user/user/g' > "$temp_file"
            
            # 比较文件是否有变化
            if ! cmp -s "$file" "$temp_file"; then
                mv "$temp_file" "$file"
                echo "Modified file: $file"
            else
                rm "$temp_file"
            fi
        fi
    done
}

# 主程序
echo "Starting directory renaming..."
rename_directories

echo "Starting file content modification..."
modify_file_contents

echo "Process completed."
