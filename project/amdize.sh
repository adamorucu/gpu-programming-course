for file in *.cu; do
    hipify-perl "$file" > "${file%.cu}.cpp"
done
