package org.example;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.PDPageContentStream.AppendMode;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.font.PDFont;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.font.Standard14Fonts.FontName;
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class PdfExample {

    private PdfExample() {}

    public static void main(String[] args) throws IOException {
        String output = "build/hello.pdf";
        String url = "https://resources.djl.ai/images/kitten.jpg";
        Path file = Paths.get("build/kitten.jpg");
        if (Files.notExists(file)) {
            try (InputStream is = new URL(url).openStream()) {
                Files.copy(is, file);
            }
        }

        try (PDDocument doc = new PDDocument()) {
            PDPage page = new PDPage();
            doc.addPage(page);
            PDFont font = new PDType1Font(FontName.HELVETICA_BOLD);
            try (PDPageContentStream contents = new PDPageContentStream(doc, page)) {
                contents.beginText();
                contents.setFont(font, 12);
                contents.newLineAtOffset(100, 700);
                contents.showText("Hello");
                contents.endText();
            }

            page = new PDPage();
            doc.addPage(page);

            PDImageXObject image = PDImageXObject.createFromFile(file.toString(), doc);
            float width = image.getWidth();
            float height = image.getHeight();
            PDRectangle rect = page.getMediaBox();
            float pageHeight = rect.getHeight();
            float pageWidth = rect.getHeight();
            float scale = Math.min(pageWidth / width, pageHeight / height);
            if (scale < 1) {
                width *= scale;
                height *= scale;
            }
            try (PDPageContentStream ps =
                    new PDPageContentStream(doc, page, AppendMode.APPEND, true, true)) {
                ps.drawImage(image, 20, pageHeight - height - 20, width, height);
            }
            Files.deleteIfExists(Paths.get(output));
            doc.save(output);
        }
    }
}
