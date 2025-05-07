package org.example;

import com.itextpdf.io.image.ImageDataFactory;
import com.itextpdf.kernel.pdf.PdfDocument;
import com.itextpdf.kernel.pdf.PdfWriter;
import com.itextpdf.layout.Document;
import com.itextpdf.layout.element.AreaBreak;
import com.itextpdf.layout.element.Image;
import com.itextpdf.layout.element.Paragraph;

import java.io.IOException;

public class ItextExample {

    public static void main(String[] args) throws IOException {
        String output = "build/hello.pdf";
        try (Document document = new Document(new PdfDocument(new PdfWriter(output)))) {
            String file = "https://resources.djl.ai/images/kitten.jpg";
            Image image = new Image(ImageDataFactory.create(file));
            document.add(new Paragraph("Hello!"));
            document.add(new AreaBreak());
            document.add(image);
        }
    }
}
