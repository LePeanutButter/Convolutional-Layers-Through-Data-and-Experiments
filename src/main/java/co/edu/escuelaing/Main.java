package co.edu.escuelaing;

import co.edu.escuelaing.config.ServerConfig;
import co.edu.escuelaing.handlers.RouteRegistry;
import co.edu.escuelaing.server.HttpServer;

/**
 * Punto de entrada de la aplicación.
 * Configura rutas, archivos estáticos e inicia el servidor HTTP.
 */
public class Main {

    public static void main(String[] args) {
        configureRoutes();
        startServer();
    }

    private static void configureRoutes() {
        RouteRegistry.staticfiles("webroot");

        RouteRegistry.get("/App/hello", (req, resp) -> {
            String name = req.getValues("name");
            return "Hello " + (name != null ? name : "World");
        });

        RouteRegistry.get("/App/pi", (req, resp) ->
                String.valueOf(Math.PI)
        );
    }

    private static void startServer() {
        ServerConfig config = ServerConfig.withPort(8080);
        HttpServer server = new HttpServer(config);
        server.start();
    }
}
