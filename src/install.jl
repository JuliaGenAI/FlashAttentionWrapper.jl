using CondaPkg

function install()
    CondaPkg.add_channel("conda-forge")
    CondaPkg.add_channel("pytorch")
    CondaPkg.add("pytorch"; channel="pytorch")

    CondaPkg.withenv() do
        pip = CondaPkg.which("pip")
        run(`$pip install flash-attn --no-build-isolation`)
    end
end