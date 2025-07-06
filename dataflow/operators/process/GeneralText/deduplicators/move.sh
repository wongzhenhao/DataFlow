#!/bin/bash

# 设置a和b文件夹路径
a_dir="./"
b_dir="/mnt/public/data/mzm/DataFlow-421/dataflow/process/text/deduplicators/"

# 确保目录存在
if [[ ! -d "$a_dir" || ! -d "$b_dir" ]]; then
  echo "文件夹 $a_dir 或 $b_dir 不存在"
  exit 1
fi

# 获取a文件夹中所有文件名（不含路径）
a_files=$(find "$a_dir" -type f -exec basename {} \;)

# 将a文件名列表存入数组
declare -A a_file_map
for name in $a_files; do
  a_file_map["$name"]=1
done

# 遍历b中的所有文件名（不含路径）
find "$b_dir" -type f | while read -r b_file; do
  filename=$(basename "$b_file")

  # 如果a中没有这个文件名，则在a中创建
  if [[ -z "${a_file_map[$filename]}" ]]; then
    new_file="$a_dir/$filename"
    echo "创建文件: $new_file"
    touch "$new_file"
  fi
done