package com.amazonaws.awscurl;

import java.io.OutputStream;

/** A OutputStream class that does nothing. */
public class NullOutputStream extends OutputStream {

    public static final NullOutputStream INSTANCE = new NullOutputStream();

    /** {@inheritDoc} */
    @Override
    public void write(int b) {}

    /** {@inheritDoc} */
    @Override
    public void write(byte[] b) {}

    /** {@inheritDoc} */
    @Override
    public void write(byte[] b, int off, int len) {}
}
