# Rascal

A language inspired by Pascal and Rust

## Inspiration

My high level goal was to develop a language that makes writing an imperative style of code
feel more sound. I haven't fully developed the language constructs that will make this possible
but I have a number of rough sources of inspirations.

## Example

This is an incredibly simple example of calculating the `n`'th Fibonacci sequence.
Currently the only way to observe output is via the return code of the program.

```
fun fib(n: int32) -> int32 begin
    if n == 0 then
        return 0;
    else if n == 1 then
        return 1;
    end

    return fib(n - 1) + fib(n - 2);
end

program calc_fib begin
    let index = 7;
    return fib(index);
end
```

## Usage

You can try the latest version of the compiler with using some basic `gcc` like options.
Build the Rascal compiler `rascalc` with Cargo and ensure you have
`gcc` installed. Then you can compile the `calc_fib` example above by running:

```
rascalc fib.ras -o fib
```

## Roadmap

I'm currently developing the basics of the language. My original goal was to target WASM,
however in an effort to speed up language development and hopefully ensure that Rascal isn't just
"high level" WASM, I'm starting by generating C. Then we can call to `gcc`.

I'm hoping to bring the `backends/wasm.rs` WASM backend up to date later.
